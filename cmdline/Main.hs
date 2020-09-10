module Main where

import           Prelude
import           RIO                          (first, forM, on, zipWithM_, (<>))
import           RIO.Char                     (isAlphaNum, isSpace, isUpper,
                                               toLower)
import           RIO.Directory
import           RIO.FilePath                 (FilePath, (</>))
import           RIO.List                     (sortBy)
import qualified RIO.Text                     as T
import           RIO.Writer                   (Writer, execWriter, tell)

import           Data.Tuple.Ops               (uncons)
import           Language.Haskell.Exts
import           Options.Applicative          hiding (optional)
import           System.Log.Logger
import           Text.ParserCombinators.ReadP
import           Text.Printf                  (printf)

import           MXNet.Base.Raw

_module_ = "Main"

data Arguments = Arguments
    { output_dir :: FilePath
    }

args_spec = Arguments
         <$> strOption (long "output" <> short 'o' <> value "operators" <> metavar "OUTPUT-DIR")

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
                , simpleImport "MXNet.Base.Tensor"
                , simpleImportVars "Data.Maybe" ["catMaybes", "fromMaybe"]]
    modDataIter = Module () (Just $ ModuleHead () (ModuleName () "MXNet.Base.DataIter") Nothing Nothing) []
                  [ simpleImport "RIO"
                  , simpleImport "RIO.List"
                  , simpleImportVars "RIO.List.Partial" ["(!!)"]
                  , simpleImport "MXNet.Base.Raw"
                  , simpleImport "MXNet.Base.Spec.Operator"
                  , simpleImport "MXNet.Base.Spec.HMap"
                  , simpleImportVars "Data.Maybe" ["catMaybes", "fromMaybe"]]

getOpName :: AtomicSymbolCreator -> IO String
getOpName = fmap T.unpack . mxSymbolGetAtomicSymbolName

tensorVar = tyVarIdent "t"

makeParamInst :: String -> [ResolvedType] -> Bool -> Decl ()
makeParamInst symname typs symbolapi =
    TypeInsDecl () (tyApp (tyApp (tyCon $ unQual $ name "ParameterList") (tyPromotedStr symname)) tensorVar)
                   (tyPromotedList paramList)
  where
    paramList = map (\(name, typ1, typ2) -> tyPromotedTuple [tyPromotedStr name, tyApp typ1 typ2]) typs

data GenFlag = GenSymbolOp
    | GenNDArrayReturn
    | GenNDArrayUpdate

makeSignature :: String -> [Asst ()]-> Decl ()
makeSignature symname extra_constraints =
    let cxfullfill = appA (name "Fullfilled") [tyPromotedStr symname, tensorVar, tyVarIdent "a"]
        cxtensor   = appA (name "Tensor") [tensorVar]
        cx_all = cxTuple (cxtensor : cxfullfill : extra_constraints)
        fun = tyFun (let hmap = tyCon $ unQual $ name "ArgsHMap"
                     in hmap `tyApp` tyPromotedStr symname
                             `tyApp` tensorVar
                             `tyApp` tyVarIdent "a")
                    (tyApp (tyCon $ unQual $ name "TensorApply") tensorVar)
        vars = [unkindedVar (name "a"), unkindedVar (name "t")]
    in tySig [name symname] $ tyForall vars cx_all fun

genTensorOp :: AtomicSymbolCreator -> IO [Decl ()]
genTensorOp sc = do
    (symname_t, _, argname_t, argtype_t, _, key_var_num_args_t, rettyp) <- mxSymbolGetAtomicSymbolInfo sc
    let symname = T.unpack symname_t
        argname = map T.unpack argname_t
        argtype = map T.unpack argtype_t
        key_var_num_args = T.unpack key_var_num_args_t

    if symname `elem` ["_Native", "_NDArray"] then
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
            let paramListInst = makeParamInst symname_ (scalarTypes ++ tensorKeyTypes ++ tensorVarTypes) True
                sig = makeSignature symname_ $
                        -- the "Custom" is a little special, because it allow extra arguments
                        if symname == "Custom"
                            --    PopKey (ArgOf "_Custom(symbol)") args "data",
                            --    Dump (PopResult (ArgOf "_Custom(symbol)") args "data"))
                            then let argOfCustom = tyParen $ (tyCon $ unQual $ name "ArgOf") `tyApp`
                                                             (tyPromotedStr "_Custom") `tyApp`
                                                             tensorVar
                                 in [appA (name "PopKey") [argOfCustom, tyVarIdent "a", tyPromotedStr "data"],
                                     appA (name "Dump") [ tyParen $
                                        (tyCon $ unQual $ name "PopResult") `tyApp`
                                        argOfCustom `tyApp`
                                        tyVarIdent "a" `tyApp`
                                        tyPromotedStr "data"]]
                            else []
            let fun = sfun (name symname_) [name "args"] (UnGuardedRhs () body) Nothing
                make_scalar_values =
                    if symname == "Custom"
                        -- dump (pop args #data)
                        then function "dump" `app` (
                                function "pop" `app`
                                (var $ name "args") `app`
                                OverloadedLabel () "data")
                        -- catMaybes
                        --    [("KEY",) . showValue <$> (args !? #KEY :: Maybe VALUE_TYPE),
                        --     ... ]
                        else function "catMaybes" `app` listE
                                ([ infixApp (infixApp (tupleSection [Just $ strE argkey, Nothing])
                                                     (op $ sym ".")
                                                     (function "showValue"))
                                           (op $ sym "<$>") $
                                               ExpTypeSig () (infixApp (var $ name "args")
                                                                   (op $ sym "!?")
                                                                   (OverloadedLabel () argkey))
                                                      (tyApp (tyCon $ unQual $ name "Maybe") typ)
                                 | (argkey, _, typ) <- scalarTypes] ++
                                 if null key_var_num_args || null tensorVarTypes || key_var_num_args `elem` [k | (k,_,_) <- scalarTypes]
                                 then []
                                 else [(con $ unQual $ name "Just") `app` tuple [
                                    strE key_var_num_args,
                                    function "showValue" `app` (function "length" `app` (var $ name "tensorVarArgs"))]])
                body =  letE ([
                          patBind (pvar $ name "scalarArgs") make_scalar_values
                        , patBind (pvar $ name "tensorKeyArgs") (function "catMaybes"
                            `app` listE [
                                infixApp (tupleSection [Just $ strE argkey, Nothing]) (op $ sym "<$>") $
                                ExpTypeSig ()
                                    (infixApp (var $ name "args") (op $ sym "!?") (OverloadedLabel () argkey))
                                    (tyApp (tyCon $ unQual $ name "Maybe") typ) | (argkey, _, typ) <- tensorKeyTypes])
                        ] ++
                        case tensorVarTypes of
                          [(argkey,_,_)] ->
                              [patBind (pvar $ name "tensorVarArgs") $ function "fromMaybe"
                                `app` eList
                                `app` ExpTypeSig ()
                                    (infixApp (var $ name "args") (op $ sym "!?") (OverloadedLabel () argkey))
                                    (tyApp (tyCon $ unQual $ name "Maybe") (tyList tensorVar))]
                          _ -> [])
                        (function "apply"
                            `app` (strE symname)
                            `app` (var $ name "scalarArgs")
                            `app` (if null tensorVarTypes
                                      then function "Left" `app` (var $ name "tensorKeyArgs")
                                      else function "Right" `app` (var $ name "tensorVarArgs")))
            return [paramListInst, sig, fun]

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
                handleTensor (ParamDescItem "NDArray")             = tensor_karg $ tensorVar
                handleTensor (ParamDescItem "NDArray-or-Symbol")   = tensor_karg $ tensorVar
                handleTensor (ParamDescItem "NDArray-or-Symbol[]") = tensor_varg $ tyList $ tensorVar
                handleTensor (ParamDescItem "Symbol")              = tensor_karg $ tensorVar
                handleTensor (ParamDescItem "Symbol[]")            = tensor_varg $ tyList $ tensorVar
                handleTensor (ParamDescItem "Symbol or Symbol[]")  = tensor_varg $ tyList $ tensorVar
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

appA = AppA ()

tupleSection = TupleSection () Boxed

con = Con ()

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
