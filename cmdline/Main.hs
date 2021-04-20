{-# LANGUAGE LambdaCase #-}
module Main where

import           Prelude
import           RIO                          (Text, first, forM, on, traceShow,
                                               zipWithM_, (<>))
import           RIO.Char                     (isAlphaNum, isSpace, isUpper,
                                               toLower)
import           RIO.Directory
import           RIO.FilePath                 (FilePath, (</>))
import           RIO.List                     (nub, sortBy)
import qualified RIO.Text                     as T
import           RIO.Writer                   (Writer, execWriter, tell)

import           Data.Bifunctor               (bimap)
import           Data.Tuple.Ops               (uncons)
import           Language.Haskell.Exts
import           Options.Applicative          hiding (optional)
import           System.Log.Logger
import           Text.ParserCombinators.ReadP
import           Text.Printf                  (printf)

import           MXNet.Base.Raw.Common
import           MXNet.Base.Raw.DataIter
import           MXNet.Base.Raw.NDArray
import           MXNet.Base.Raw.Symbol

_module_ = "Main"

data Arguments = Arguments
    { output_dir :: FilePath
    }

args_spec = Arguments
         <$> strOption (long "output" <> short 'o' <> value "tmp" <> metavar "OUTPUT-DIR")

main = do
    updateGlobalLogger _module_ (setLevel INFO)
    args <- execParser opts
    let base = output_dir args </> "MXNet" </> "Base" </> "Operators"
    createDirectoryIfMissing True base

    ops  <- mxSymbolListAtomicSymbolCreators
    op_names <- mapM getOpName ops
    let ops_sorted = map snd $ sortBy (compare `on` fst) $ zip op_names ops

    infoM _module_ "Generating NDArray/Symbol operators..."
    ops <- concat <$> mapM genTensorOp ops_sorted
    writeFile (base </> "Tensor.hs") $ prettyPrint (modTensor ops)

    dataitercreators  <- mxListDataIters
    infoM _module_ "Generating DataIter..."
    dataiters <- concat <$> mapM genDataIter (zip dataitercreators [0..])
    let base = output_dir args </> "MXNet" </> "Base"
    writeFile (base </> "DataIter.hs") $ prettyPrint (modDataIter dataiters)

  where
    opts = info (args_spec <**> helper) (fullDesc <> progDesc "Generate MXNet operators")
    modTensor = Module () (Just $ ModuleHead () (ModuleName () "MXNet.Base.Operators.Tensor") Nothing Nothing) []
                [ simpleImport "RIO"
                , simpleImport "RIO.List"
                , simpleImport "MXNet.Base.Raw"
                , simpleImport "MXNet.Base.Spec.Operator"
                , simpleImport "MXNet.Base.Spec.HMap"
                , simpleImport "MXNet.Base.Tensor.Class"
                , simpleImportVars "MXNet.Base.Types" ["DType"]
                , simpleImportVars "Data.Maybe" ["catMaybes", "fromMaybe"]]
    modDataIter = Module () (Just $ ModuleHead () (ModuleName () "MXNet.Base.DataIter") Nothing Nothing) []
                  [ simpleImport "RIO"
                  , simpleImport "RIO.List"
                  , simpleImportVars "RIO.List.Partial" ["(!!)"]
                  , simpleImportVars "MXNet.Base.Types" ["DType"]
                  , simpleImport "MXNet.Base.Raw"
                  , simpleImport "MXNet.Base.Spec.Operator"
                  , simpleImport "MXNet.Base.Spec.HMap"
                  , simpleImportVars "Data.Maybe" ["catMaybes", "fromMaybe"]]

getOpName :: AtomicSymbolCreator -> IO String
getOpName = fmap T.unpack . mxSymbolGetAtomicSymbolName

tensorVar = tyVarIdent "t"

makeParamInst :: String -> [Type ()] -> [ResolvedType] -> Bool -> Decl ()
makeParamInst symname dvars argtyps symbolapi =
    TypeInsDecl () (tyApp (tyApp (tyCon $ unQual $ name "ParameterList") (tyPromotedStr symname)) pparm)
                   (tyPromotedList paramList)
    where
        tvar      = tyVarIdent "t"
        pparm     = tyPromotedTuple $ case dvars of
                      [] -> []
                      _  -> tvar:dvars
        paramList = map (\(name, typ1, typ2) -> tyPromotedTuple [tyPromotedStr name, tyApp typ1 typ2]) argtyps

data GenFlag = GenSymbolOp
    | GenNDArrayReturn
    | GenNDArrayUpdate

makeSignature :: String -> [Type ()] -> Type () -> [Asst ()] -> Decl ()
makeSignature symname din dout extra_constraints =
    let dtype_vars = nub (gatherDtyVarsFromType dout ++ din)
        hargs_var  = tyVarIdent "a"
        tensor_var = tyVarIdent "t"
        pparm      = tyPromotedTuple $ case din of
                       [] -> []
                       _  -> tensor_var:din
        cxfullfill = appA (name "Fullfilled") [tyPromotedStr symname, pparm, hargs_var]
        cxtensor   = appA (name "Tensor") [tensor_var]
        cxdtype    = map (\v -> appA (name "DType") [v]) dtype_vars
        callstack  = appA (name "HasCallStack") []
        cx_all     = cxTuple (cxtensor : cxfullfill : callstack : (cxdtype ++ extra_constraints))
        in_type    = let hmap = tyCon $ unQual $ name "ArgsHMap"
                      in hmap `tyApp` tyPromotedStr symname
                             `tyApp` pparm
                             `tyApp` hargs_var
        out_type   = (tyCon $ unQual $ name "TensorApply") `tyApp` dout
        fun        = tyFun in_type out_type
        dtype_var_names = map (\case {TyVar () (Ident () n) -> n}) dtype_vars
        vars       = map (unkindedVar . name) $ "a" : "t" : dtype_var_names
    in tySig [name symname] $ tyForall vars cx_all fun

patchOutputTensor :: String -> [Type ()] -> Type ()
patchOutputTensor symname input_dtype_var
  | symname == "Cast" = tensor_var `tyApp` tyVarIdent "v"
  | symname `elem` npi_boolean_ops = tensor_var `tyApp` (tyCon $ unQual $ name "Bool")
  | symname `elem` npi_argminmax   = tensor_var `tyApp` (tyCon $ unQual $ name "Int64")
  | otherwise = case input_dtype_var of
                  []  -> tensor_var `tyApp` tyVarIdent "u"
                  [a] -> tensor_var `tyApp` a
                  _   -> error ("multiple input dtypes " ++ show input_dtype_var)
    where
        tensor_var = tyVarIdent "t"
        npi_boolean_ops = [ "_npi_equal"
                          , "_npi_not_equal"
                          , "_npi_greater"
                          , "_npi_less"
                          , "_npi_greater_equal"
                          , "_npi_less_equal"
                          , "_npi_equal_scalar"
                          , "_npi_not_equal_scalar"
                          , "_npi_greater_scalar"
                          , "_npi_less_scalar"
                          , "_npi_greater_equal_scalar"
                          , "_npi_less_equal_scalar"
                          ]
        npi_argminmax = ["_npi_argmax", "_npi_argmin"]

patchTensorKeyArgs :: String -> [ResolvedType] -> [ResolvedType]
patchTensorKeyArgs symname keyargs
  | symname == "_npi_where" = map fixup keyargs
  | otherwise = keyargs
    where
        tvar = tyVarIdent "t"
        bool = tyCon $ unQual $ name "Bool"
        fixup ("condition", attr, _) = ("condition", attr, tvar `tyApp` bool)
        fixup x                      = x

patchConstraints :: String -> [Type ()] -> [Asst ()]
patchConstraints symname input_dtype_var
  | symname == "Custom" = -- the "Custom" is a little special, because it allow extra arguments
                          --    PopKey (ArgOf "_Custom(symbol)") args "data",
                          --    Dump (PopResult (ArgOf "_Custom(symbol)") args "data"))
                          let tvar = tyVarIdent "t"
                              argOfCustom = tyParen $ (tyCon $ unQual $ name "ArgOf") `tyApp`
                                             (tyPromotedStr "_Custom") `tyApp`
                                             (tyPromotedTuple (tvar:input_dtype_var))
                           in [appA (name "PopKey") [argOfCustom, tyVarIdent "a", tyPromotedStr "data"],
                               appA (name "Dump") [ tyParen $
                               (tyCon $ unQual $ name "PopResult") `tyApp`
                               argOfCustom `tyApp`
                               tyVarIdent "a" `tyApp`
                               tyPromotedStr "data"]]
  | otherwise = []

genTensorOp :: AtomicSymbolCreator -> IO [Decl ()]
genTensorOp sc = do
    (symname_t, _, argname_t, argtype_t, _, key_var_num_args_t, rettyp) <- mxSymbolGetAtomicSymbolInfo sc
    let symname = T.unpack symname_t
        (argname, argtype) = bimap (map T.unpack) (map T.unpack) $ numpyFixups symname_t argname_t argtype_t
        key_var_num_args = T.unpack key_var_num_args_t
        unsupportedOps = ["_Native", "_NDArray", "_CrossDeviceCopy",
                          "_CustomFunction", "_FusedOpHelper", "_FusedOpOutHelper"]

    if symname `elem` unsupportedOps then
        return []
    else do
        let symname_ = normalizeName symname
            (errs, scalarTypes, tensorKeyTypes, tensorVarTypes) = execWriter $ zipWithM_ (resolveHaskellType ResolveTensor) argname argtype

        if not (null errs) then do
            forM errs $ \(name, msg) ->
                errorM _module_ (printf "Function: %s %s" symname msg)
            return []
        else if (length tensorVarTypes >= 2) then do
            errorM _module_ (printf "Function: %s has more than one NDArray/Symbol[] argument." symname)
            return []
        else if (not (null tensorKeyTypes) && not (null tensorVarTypes)) then do
            errorM _module_ (printf "Function: %s has both NDArray/Symbol and NDArray/Symbol[] arguments." symname)
            return []
        else do
            let tensorKeyArgs = patchTensorKeyArgs symname tensorKeyTypes
                dvars         = gatherDtyVars (tensorKeyArgs ++ tensorVarTypes)
                outTensor     = patchOutputTensor  symname dvars
                extraCst      = patchConstraints   symname dvars
                paramListInst = makeParamInst symname_ dvars (scalarTypes ++ tensorKeyArgs ++ tensorVarTypes) True
                sig           = makeSignature symname_ dvars outTensor extraCst
                fun           = sfun (name symname_) [name "args"] (UnGuardedRhs () body) Nothing
                scalarArgs    = -- catMaybes
                                --    [("KEY",) . showValue <$> (args !? #KEY :: Maybe VALUE_TYPE),
                                --     ... ]
                                function "catMaybes" `app`
                                listE ([infixApp (infixApp (tupleSection [Just $ strE argkey, Nothing])
                                                           (op $ sym ".")
                                                           (function "showValue"))
                                                 (op $ sym "<$>") $
                                                     ExpTypeSig ()
                                                        (infixApp
                                                            (var $ name "args")
                                                            (op $ sym "!?")
                                                            (OverloadedLabel () argkey))
                                                        (tyApp (tyCon $ unQual $ name "Maybe") typ)
                                       | (argkey, _, typ) <- scalarTypes] ++
                                            if null key_var_num_args ||
                                               null tensorVarTypes   ||
                                               key_var_num_args `elem` [k | (k,_,_) <- scalarTypes]
                                            then []
                                            else [(con $ unQual $ name "Just") `app`
                                                  tuple [ strE key_var_num_args
                                                        , app (function "showValue") $
                                                          app (function "length")    $
                                                          var $ name "tensorVarArgs"
                                                        ]])
                scalarArgsForCustom = -- dump (pop args #data)
                                      app (function "dump") $
                                      function "pop" `app` (var $ name "args") `app` OverloadedLabel () "data"
                tvar = tyVarIdent "t"
                raw_tensor = tyApp (tyCon $ unQual $ name "RawTensor") tvar
                body = letE ([
                         patBind (pvar $ name "scalarArgs")
                                 (if symname == "Custom" then scalarArgsForCustom else scalarArgs)
                       , patBind (pvar $ name "tensorKeyArgs") (function "catMaybes"
                           `app` listE [
                               infixApp
                                   (infixApp (tupleSection [Just $ strE argkey, Nothing])
                                             (op $ sym ".")
                                             (var $ name "toRaw"))
                                   (op $ sym "<$>") $
                                   ExpTypeSig ()
                                       (infixApp (var $ name "args") (op $ sym "!?") (OverloadedLabel () argkey))
                                       (tyApp (tyCon $ unQual $ name "Maybe") typ) | (argkey, _, typ) <- tensorKeyArgs])
                       ] ++
                       case tensorVarTypes of
                         [(argkey,_,_)] ->
                             [patBind (pvar $ name "tensorVarArgs") $ function "fromMaybe"
                               `app` eList
                               `app` ExpTypeSig ()
                                   (infixApp ((var $ name "map") `app` (var $ name "toRaw"))
                                             (op $ sym "<$>")
                                             (infixApp (var $ name "args")
                                                       (op $ sym "!?")
                                                       (OverloadedLabel () argkey)))
                                   (tyApp (tyCon $ unQual $ name "Maybe") (tyList raw_tensor))]
                         _ -> [])
                       (function "applyRaw"
                           `app` (strE symname)
                           `app` (var $ name "scalarArgs")
                           `app` (if null tensorVarTypes
                                     then function "Left" `app` (expTypeSig
                                           (var $ name "tensorKeyArgs")
                                           (tyList $ tyTuple [tyCon $ unQual $ name "Text", raw_tensor]))
                                     else function "Right" `app` (var $ name "tensorVarArgs")))
            return $ [paramListInst, sig, fun]

genDataIter :: (DataIterCreator, Integer) -> IO [Decl ()]
genDataIter (dataitercreator, index) = do
    (diname_t, _, argnames_t, argtypes_t, _) <- mxDataIterGetIterInfo dataitercreator
    let diname = normalizeName $ T.unpack diname_t
        argnames = map T.unpack argnames_t
        argtypes = map T.unpack argtypes_t

        (errs, scalarTypes, _, _) = execWriter $ zipWithM_ (resolveHaskellType ResolveDataIter) argnames argtypes

        -- parameter list
        paramList = map (\(name, typ1, typ2) -> tyPromotedTuple [tyPromotedStr name, tyApp typ1 typ2]) scalarTypes
        paramInst = TypeInsDecl ()
                        (tyApp (tyApp (tyCon $ unQual $ name "ParameterList") (tyPromotedStr diname)) (tyVarIdent "dummy"))
                        (tyPromotedList paramList)

        -- signature
        void = tyTuple []
        cxfullfill = appA (name "Fullfilled") [tyPromotedStr diname, void, tyVarIdent "a"]
        tyfun = tyFun
                    (let hmap = tyCon $ unQual $ name "ArgsHMap"
                     in hmap `tyApp` tyPromotedStr diname `tyApp` void `tyApp` tyVarIdent "a")
                    (tyApp (tyCon $ unQual $ name "IO") (tyCon $ unQual $ name "DataIterHandle"))
        tysig = tySig [name diname] $ tyForall [unkindedVar (name "a")] (cxSingle cxfullfill) tyfun

        -- function
        fun = sfun (name diname) [name "args"] (UnGuardedRhs () body) Nothing
        body = letE ([
                patBind (pvar $ name "allargs") (function "catMaybes"
                    `app` listE [
                        infixApp (infixApp (tupleSection [Just $ strE argkey, Nothing]) (op $ sym ".") (function "showValue")) (op $ sym "<$>") $
                            ExpTypeSig () (infixApp (var $ name "args") (op $ sym "!?") (OverloadedLabel () argkey)) (tyApp (tyCon $ unQual $ name "Maybe") typ) | (argkey, _, typ) <- scalarTypes])
              , patBind (pTuple [pvar $ name "keys", pvar $ name "vals"]) (app (function "unzip") $ var $ name "allargs")
              ]) (doE $ [
                  genStmt (pvar $ name "dis") $ function "mxListDataIters",
                  genStmt (pvar $ name "di") $ function "return" `app` (infixApp (var $ name "dis") (op $ sym "!!") (intE index)),
                  qualStmt $ function "mxDataIterCreateIter" `app` (var $ name "di") `app` (var $ name "keys") `app` (var $ name "vals")
              ])

    if not (null errs) then do
       forM errs $ \(name, msg) ->
           errorM _module_ (printf "Function: %s %s" diname msg)
       return []
    else
       return [paramInst, tysig, fun]

normalizeName :: String -> String
normalizeName name = '_': name

uncapitalize :: String -> String
uncapitalize []     = []
uncapitalize (x:xs) = toLower x: xs

numpyFixups :: Text -> [Text] -> [Text] -> ([Text], [Text])
numpyFixups symname argnames argtypes = unzip $ zipWith (fixup symname) argnames argtypes
    where
        fixup "_npi_concatenate" "dim" t = ("axis", t)
        fixup _ n t                      = (n, t)

data ParamDesc = ParamDescItem String
    | ParamDescList Bool [String]
    deriving (Eq, Show)
data ResolveMode = ResolveTensor
    | ResolveDataIter
    deriving Eq

type ResolvedType = (String, Type (), Type ())
resolveHaskellType :: ResolveMode -> String -> String -> Writer ([(String, String)], [ResolvedType], [ResolvedType], [ResolvedType]) ()
resolveHaskellType mode symname desc = do
    let fail   msg   = tell ([(symname, msg)], [], [], [])
    case readP_to_S typedesc desc of
        [(fields, "")] -> do
            let required = ParamDescItem "required" `elem` fields
                attr = tyCon $ unQual $ name $ if required then "AttrReq" else "AttrOpt"
                scalar hstyp      = tell ([], [(symname_, attr, hstyp)], [], [])
                tensor_karg hstyp = tell ([], [], [(symname_, attr, hstyp)], [])
                tensor_varg hstyp = tell ([], [], [], [(symname_, attr, hstyp)])
                symname_ = uncapitalize symname
                --
                -- tensor operators
                --
                tensor_var = tyVarIdent "t"
                dtype_var  = tyVarIdent "u"
                tensor_ty  = tensor_var `tyApp` dtype_var
                handleTensor (ParamDescItem "NDArray")             = tensor_karg $ tensor_ty
                handleTensor (ParamDescItem "NDArray-or-Symbol")   = tensor_karg $ tensor_ty
                handleTensor (ParamDescItem "NDArray-or-Symbol[]") = tensor_varg $ tyList $ tensor_ty
                handleTensor (ParamDescItem "Symbol")              = tensor_karg $ tensor_ty
                handleTensor (ParamDescItem "Symbol[]")            = tensor_varg $ tyList $ tensor_ty
                handleTensor (ParamDescItem "Symbol or Symbol[]")  = tensor_varg $ tyList $ tensor_ty
                handleTensor t = fallThrough t

                fallThrough t = fail $ printf "Unknown type: arg %s(%s)." symname desc

            case head fields of
                ParamDescItem "Shape(tuple)"        -> scalar $ tyList $ tyCon $ unQual $ name "Int"
                ParamDescItem "int"                 -> scalar $ tyCon $ unQual $ name "Int"
                ParamDescItem "int (non-negative)"  -> scalar $ tyCon $ unQual $ name "Int"
                ParamDescItem "long (non-negative)" -> scalar $ tyCon $ unQual $ name "Int"
                ParamDescItem "boolean"             -> scalar $ tyCon $ unQual $ name "Bool"
                ParamDescItem "float"               -> scalar $ tyCon $ unQual $ name "Float"
                ParamDescItem "double"              -> scalar $ tyCon $ unQual $ name "Double"
                ParamDescItem "float32"             -> scalar $ tyCon $ unQual $ name "Float"
                -- real_t (from mshadow) is by default float.
                ParamDescItem "real_t"              -> scalar $ tyCon $ unQual $ name "Float"
                ParamDescItem "string"              -> scalar $ tyCon $ unQual $ name "Text"
                ParamDescItem "int or None"         -> scalar $ tyApp (tyCon $ unQual $ name "Maybe") (tyCon $ unQual $ name "Int")
                ParamDescItem "float or None"       -> scalar $ tyApp (tyCon $ unQual $ name "Maybe") (tyCon $ unQual $ name "Float")
                ParamDescItem "double or None"      -> scalar $ tyApp (tyCon $ unQual $ name "Maybe") (tyCon $ unQual $ name "Double")
                ParamDescItem "boolean or None"     -> scalar $ tyApp (tyCon $ unQual $ name "Maybe") (tyCon $ unQual $ name "Bool")
                ParamDescItem "Shape or None"       -> scalar $ tyApp (tyCon $ unQual $ name "Maybe") (tyList $ tyCon $ unQual $ name "Int")
                ParamDescItem "tuple of <float>"    -> scalar $ tyList $ tyCon $ unQual $ name "Float"
                ParamDescItem "tuple of <double>"   -> scalar $ tyList $ tyCon $ unQual $ name "Double"
                ParamDescList hasnone vs -> do
                    let vsprom = map tyPromotedStr vs
                        typ1 = tyApp (tyCon $ unQual $ name "EnumType") (tyPromotedList vsprom)
                        typ2 = tyApp (tyCon $ unQual $ name "Maybe") typ1
                    scalar $ if hasnone then typ2 else typ1
                t | mode == ResolveTensor -> handleTensor t
                  | mode == ResolveDataIter -> fallThrough t
        other -> fail (printf "cannot parse type description: %s. Result: %s" desc (show other))

typedesc = do
    -- since 1.3, there are types starting with ',', and it implies 'int' type.
    def <- (char ',' >> return [ParamDescItem "int"]) <++ return []
    ds <- sepBy (skipSpaces >> (list1 +++ list2 +++ item)) (char ',')
    eof
    return $ def ++ ds
  where
    list1 = ParamDescList True  <$> between (string "{None,") (char '}') (sepBy (skipSpaces >> strItem) (char ','))
    list2 = ParamDescList False <$> between (string "{") (char '}') (sepBy (skipSpaces >> strItem) (char ','))
    strItem = between (char '\'') (char '\'') (munch1 (\c -> isAlphaNum c || oneOf c "_-+-./"))
    item = ParamDescItem <$> munch1 (\c -> isAlphaNum c || oneOf c " _-+()=[]<>./'")
    oneOf :: Eq a => a -> [a] -> Bool
    oneOf c wl = c `elem` wl

gatherDtyVars :: [ResolvedType] -> [Type ()]
gatherDtyVars = nub . concatMap (\ (_, _, ty) -> gatherDtyVarsFromType ty)

gatherDtyVarsFromType = walk
    where
        walk v@(TyVar _ _)   = [v]
        walk (TyParen _ u)   = walk u
        walk (TyTuple _ _ u) = concatMap walk u
        walk (TyList _ u)    = walk u
        walk (TyApp _ _ v)   = walk v
        walk (TyCon _ _)     = []
        walk v               = error ("the case should not happen: " ++ show v)


unQual = UnQual ()
unkindedVar = UnkindedVar ()

tyCon = TyCon ()
tyVarSymbol = TyVar () . Symbol ()
tyVarIdent = TyVar () . Ident ()
tyApp = TyApp ()
tyFun = TyFun ()
tySig names types = TypeSig () names types
tyList = TyList ()
tyTuple = TyTuple () Boxed
tyVar = TyVar ()
tyParen = TyParen ()

tyPromotedInteger s = TyPromoted () (PromotedInteger () s (show s))
tyPromotedStr s     = TyPromoted () (PromotedString () s s)
tyPromotedList s    = TyPromoted () (PromotedList () True s)
tyPromotedTuple s   = TyPromoted () (PromotedTuple () s)

tyForall vars cxt typ = TyForall () vars_ cxt_ typ
  where
    vars_ = if null vars then Nothing else Just vars
    cxt_  = if cxt == CxEmpty () then Nothing else Just cxt

cxSingle = CxSingle ()
cxTuple  = CxTuple ()

appA n ts = TypeA () $ foldl tyApp (tyCon $ unQual n) ts

tupleSection = TupleSection () Boxed

con = Con ()

expTypeSig = ExpTypeSig ()

simpleImport mod = ImportDecl {
    importAnn = (),
    importModule = ModuleName () mod,
    importQualified = False,
    importSrc = False,
    importSafe = False,
    importPkg = Nothing,
    importAs = Nothing,
    importSpecs = Nothing
}

simpleImportVars mod vars = ImportDecl {
    importAnn = (),
    importModule = ModuleName () mod,
    importQualified = False,
    importSrc = False,
    importSafe = False,
    importPkg = Nothing,
    importAs = Nothing,
    importSpecs = Just $ ImportSpecList () False [IVar () $ Ident () var | var <- vars]
}
