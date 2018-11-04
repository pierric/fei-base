module Main where

import Options.Applicative
import Data.Semigroup ((<>))
import Language.Haskell.Exts
import qualified Data.Text as T
import System.Log.Logger
import Control.Monad
import Control.Monad.Writer
import Data.Either
import Data.Char (toLower, isUpper, isSpace, isAlphaNum)
import Text.Printf (printf)
import Text.ParserCombinators.ReadP
import System.FilePath
import System.Directory

import MXNet.Base.Raw

_module_ = "Main"

data Arguments = Arguments {
    output_dir :: FilePath
}

args_spec = Arguments 
         <$> strOption (long "output" <> short 'o' <> value "operators" <> metavar "OUTPUT-DIR")

main = do
    updateGlobalLogger _module_ (setLevel INFO)
    args <- execParser opts
    let base = output_dir args </> "MXNet" </> "Base"
    createDirectoryIfMissing True base

    ops  <- mxSymbolListAtomicSymbolCreators

    infoM _module_ "Generating Symbol operators..."
    symbols <- concat <$> mapM genSymOp ops
    writeFile (base </> "Symbol.hs") $ prettyPrint (modSymbol symbols)

    infoM _module_ "Generating NDArray operators..."
    arrays  <- concat <$> mapM genArrOp ops
    writeFile (base </> "NDArray.hs") $ prettyPrint (modArray arrays)
    
  where
    opts = info (args_spec <**> helper) (fullDesc <> progDesc "Generate MXNet operators")
    modSymbol = Module () (Just $ ModuleHead () (ModuleName () "MXNet.Base.Symbol") Nothing Nothing) [] 
                [ simpleImport "MXNet.Base.Raw"
                , simpleImport "MXNet.Base.Operator"
                , simpleImport "MXNet.Base.HMap"
                , simpleImportVars "Data.Maybe" ["catMaybes", "fromMaybe"]]
    modArray  = Module () (Just $ ModuleHead () (ModuleName () "MXnet.Base.NDArray") Nothing Nothing) []
                [ simpleImport "MXNet.Base.Raw"
                , simpleImport "MXNet.Base.Operator"
                , simpleImport "MXNet.Base.HMap"
                , simpleImportVars "Data.Maybe" ["catMaybes", "fromMaybe"]]

genSymOp :: AtomicSymbolCreator -> IO [Decl ()]
genSymOp sc = do
    (symname, desc, argname, argtype, argdesc, key_var_num_args, rettyp) <- mxSymbolGetAtomicSymbolInfo sc
    if symname `elem` ["_Native", "_NDArray", "Custom"] then
        return []
    else do
        let symname_ = normalizeName symname
            (errs, scalarTypes, symbolTypes, arrayTypes) = execWriter $ zipWithM_ (resolveHaskellType True) argname argtype
        
        if not (null errs) then do
            forM errs $ \(name, msg) -> 
                errorM _module_ (printf "Function: %s %s" symname msg)
            return []
        else if (length arrayTypes >= 2) then do
            errorM _module_ (printf "Function: %s has more than one Symbol[] argument." symname)
            return []
        else if (not (null symbolTypes) && not (null arrayTypes)) then do
            errorM _module_ (printf "Function: %s is varadic, but it also has Symbol argument." symname)
            return []
        else if (not (null arrayTypes) && null key_var_num_args) then do
            errorM _module_ (printf "Function: %s is varadic, but the variable name for num_args is not specified." symname)
            return []
        else do
            let paramList = map (\(name, typ1, typ2) -> tyPromotedTuple [tyPromotedStr name, tyApp typ1 typ2]) (scalarTypes ++ symbolTypes ++ arrayTypes) 
                paramListInst = TypeInsDecl () 
                                    (tyApp (tyCon $ unQual $ name "ParameterList") (tyPromotedStr symname_))
                                    (tyPromotedList paramList)
                cxfullfill = appA (name "Fullfilled") [tyPromotedStr symname_, tyVarIdent "args"]

                typ = tyForall [unkindedVar (name "args")]
                        (cxSingle cxfullfill)
                        (tyFun 
                          (tyCon $ unQual $ name "String")
                          (tyFun 
                            (tyApp (tyApp (tyCon $ unQual $ name "ArgsHMap") (tyPromotedStr symname_)) (tyVarIdent "args")) 
                            (tyApp (tyCon $ unQual $ name "IO") (tyVarIdent "SymbolHandle"))))
                sig = tySig [name symname_] typ
    
            let fun = sfun (name symname_) [name "name", name "args"] (UnGuardedRhs () body) Nothing
                body =  letE ([
                          patBind (pvar $ name "scalarArgs") (function "catMaybes" 
                            `app` listE [
                                infixApp (infixApp (tupleSection [Just $ strE argkey, Nothing]) (op $ sym ".") (function "showValue")) (op $ sym "<$>") $ 
                                ExpTypeSig () (infixApp (var $ name "args") (op $ sym "!?") (OverloadedLabel () argkey)) (tyApp (tyCon $ unQual $ name "Maybe") typ) | (argkey, _, typ) <- scalarTypes])
                        , patBind (pTuple [pvar $ name "scalarkeys", pvar $ name "scalarvals"]) (app (function "unzip") $ var $ name "scalarArgs")
                        , patBind (pvar $ name "symbolArgs") (function "catMaybes" 
                            `app` listE [
                                infixApp (tupleSection [Just $ strE argkey, Nothing]) (op $ sym "<$>") $ 
                                ExpTypeSig () (infixApp (var $ name "args") (op $ sym "!?") (OverloadedLabel () argkey)) (tyApp (tyCon $ unQual $ name "Maybe") typ) | (argkey, _, typ) <- symbolTypes])
                        , patBind (pTuple [pvar $ name "symbolkeys", pvar $ name "symbolvals"]) (app (function "unzip") $ var $ name "symbolArgs")]
                        ++ 
                        case arrayTypes of
                          [(argkey,_,_)] -> [patBind (pvar $ name "array") $ function "fromMaybe" 
                                                `app` eList 
                                                `app` ExpTypeSig () 
                                                        (infixApp (var $ name "args") (op $ sym "!?") (OverloadedLabel () argkey)) 
                                                        (tyApp (tyCon $ unQual $ name "Maybe") (tyList $ tyCon $ unQual $ name "SymbolHandle"))]
                          _ -> [])
                        (doE $ 
                            [ genStmt (pvar $ name "op") $ function "nnGetOpHandle"`app` strE symname ] ++
                            ( if null arrayTypes then
                                  [ genStmt (pvar $ name "sym") $ function "mxSymbolCreateAtomicSymbol" 
                                      `app` (function "fromOpHandle" `app` (var $ name "op"))
                                      `app` (var $ name "scalarkeys")
                                      `app` (var $ name "scalarvals")
                                  , qualStmt $ function "mxSymbolCompose"
                                      `app` (var $ name "sym")
                                      `app` (var $ name "name")
                                      `app` ((con $ unQual $ name "Just") `app` (var $ name "symbolkeys"))
                                      `app` (var $ name "symbolvals") ]
                              else 
                                  [ genStmt (pvar $ name "sym") $ 
                                      If () (function "hasKey" `app` (var $ name "args") `app` (OverloadedLabel () key_var_num_args))
                                          (function "mxSymbolCreateAtomicSymbol" 
                                              `app` (function "fromOpHandle" `app` (var $ name "op"))
                                              `app` (var $ name "scalarkeys")
                                              `app` (var $ name "scalarvals"))
                                          (function "mxSymbolCreateAtomicSymbol" 
                                              `app` (function "fromOpHandle" `app` (var $ name "op"))
                                              `app` (infixApp (strE key_var_num_args) (QConOp () $ Special () $ Cons ()) (var $ name "scalarkeys"))
                                              `app` (infixApp (function "showValue" `app` (function "length" `app` (var $ name "array")))  (QConOp () $ Special () $ Cons ()) (var $ name "scalarvals")))
                                  , qualStmt $ function "mxSymbolCompose"
                                      `app` (var $ name "sym")
                                      `app` (var $ name "name")
                                      `app` (con $ unQual $ name "Nothing")
                                      `app` (var $ name "symbolvals")] ) ++
                            [ qualStmt $ function "return" `app` (var $ name "sym") ])
            return [paramListInst, sig, fun]

genArrOp :: AtomicSymbolCreator -> IO [Decl ()]
genArrOp sc = do
    (symname, desc, argname, argtype, argdesc, key_var_num_args, rettyp) <- mxSymbolGetAtomicSymbolInfo sc
    if symname `elem` ["_Native", "_NDArray", "Custom"] then
        return []
    else do
        let symname_ = normalizeName symname
            (errs, scalarTypes, symbolTypes, arrayTypes) = execWriter $ zipWithM_ (resolveHaskellType False) argname argtype
        
        if not (null errs) then do
            forM errs $ \(name, msg) -> 
                errorM _module_ (printf "Function: %s %s" symname msg)
            return []
        else if (length arrayTypes >= 2) then do
            errorM _module_ (printf "Function: %s has more than Symbol[] argument." symname)
            return []
        else if (not (null symbolTypes) && not (null arrayTypes)) then do
            errorM _module_ (printf "Function: %s is varadic, but it also has Symbol argument." symname)
            return []
        else if (not (null arrayTypes) && null key_var_num_args) then do
            errorM _module_ (printf "Function: %s is varadic, but the variable name for num_args is not specified." symname_)
            return []
        else do
            let paramList = map (\(name, typ1, typ2) -> tyPromotedTuple [tyPromotedStr name, tyApp typ1 typ2]) (scalarTypes ++ symbolTypes ++ arrayTypes) 
                paramListInst = TypeInsDecl () 
                                    (tyApp (tyCon $ unQual $ name "ParameterList") (tyPromotedStr symname_))
                                    (tyPromotedList paramList)
                cxfullfill = appA (name "Fullfilled") [tyPromotedStr symname_, tyVarIdent "args"]

                typ = tyForall [unkindedVar (name "args")]
                        (cxSingle cxfullfill)
                        (tyFun 
                          (tyCon $ unQual $ name "String")
                          (tyFun 
                            (tyApp (tyApp (tyCon $ unQual $ name "ArgsHMap") (tyPromotedStr symname_)) (tyVarIdent "args")) 
                            (tyApp (tyCon $ unQual $ name "IO") (tyVarIdent "NDArrayHandle"))))
                sig = tySig [name symname_] typ
            return [paramListInst, sig]

normalizeName :: String -> String
normalizeName name@(c:cs) 
    | isUpper c = '_' : name
    | name == "where" = "_where"
    | otherwise = name

data ParamDesc = ParamDescItem String | ParamDescList Bool [String] deriving (Eq, Show)

type ResolvedType = (String, Type (), Type ())
resolveHaskellType :: Bool -> String -> String -> Writer ([(String, String)], [ResolvedType], [ResolvedType], [ResolvedType]) ()
resolveHaskellType asSymbol symname desc =
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
        ParamDescItem "string"              -> scalar $ tyCon $ unQual $ name "String"
        ParamDescItem "int or None"         -> scalar $ tyApp (tyCon $ unQual $ name "Maybe") (tyCon $ unQual $ name "Int")
        ParamDescItem "double or None"      -> scalar $ tyApp (tyCon $ unQual $ name "Maybe") (tyCon $ unQual $ name "Double")
        ParamDescItem "tuple of <float>"    -> scalar $ tyList $ tyCon $ unQual $ name "Float"
        ParamDescItem "tuple of <double>"   -> scalar $ tyList $ tyCon $ unQual $ name "Double"
        ParamDescList hasnone vs -> do
            let vsprom = map tyPromotedStr vs
                typ1 = tyApp (tyCon $ unQual $ name "EnumType") (tyPromotedList vsprom)
                typ2 = tyApp (tyCon $ unQual $ name "Maybe") typ1
            scalar $ if hasnone then typ2 else typ1

        t -> (if asSymbol then handleSymbol else handleNDArray) t
  where
    typedesc = do
        ds <- sepBy (skipSpaces >> (list1 +++ list2 +++ item)) (char ',')
        eof
        return ds
    list1 = ParamDescList True  <$> between (string "{None,") (char '}') (sepBy (skipSpaces >> listItem) (char ','))
    list2 = ParamDescList False <$> between (string "{") (char '}') (sepBy (skipSpaces >> listItem) (char ','))
    listItem = between (char '\'') (char '\'') (munch1 (\c -> isAlphaNum c || c `elem` "_"))
    item = ParamDescItem <$> munch1 (\c -> isAlphaNum c || c `elem` " _-()=[]<>'.")
    runP str = case readP_to_S typedesc str of 
                    [(xs, "")] -> xs
                    other -> error ("cannot parse type description: " ++ str)

    fields = runP desc
    optional = ParamDescItem "optional" `elem` fields
    symname_ = normalizeName symname
    attr = tyCon $ unQual $ name $ if optional then "AttrOpt" else "AttrReq"
    scalar hstyp = tell ([], [(symname_, attr, hstyp)], [], [])
    symbol hstyp = tell ([], [], [(symname_, attr, hstyp)], [])
    array  hstyp = tell ([], [], [], [(symname_, attr, hstyp)])
    fail   msg   = tell ([(symname, msg)], [], [], [])
    --
    -- symbol operators
    --
    -- operator can have only one argument of the type symbol or ndarray array
    -- and not having any other argument of symbol or ndarray
    -- besides the operator's info must definitely have a key_var_num_args, which
    -- indicates an (might be additional) argument which should be passed to mxSymbolCreateAtomicSymbol.
    handleSymbol (ParamDescItem "Symbol")              = symbol $ tyCon $ unQual $ name "SymbolHandle"
    handleSymbol (ParamDescItem "NDArray-or-Symbol")   = symbol $ tyCon $ unQual $ name "SymbolHandle"
    handleSymbol (ParamDescItem "Symbol[]")            = array $ tyList $ tyCon $ unQual $ name "SymbolHandle"
    handleSymbol (ParamDescItem "NDArray-or-Symbol[]") = array $ tyList $ tyCon $ unQual $ name "SymbolHandle"
    handleSymbol (ParamDescItem "Symbol or Symbol[]")  = array $ tyList $ tyCon $ unQual $ name "SymbolHandle"
    handleSymbol (ParamDescItem "NDArray")             = fail $ printf "NDArrayHandle arg: %s" symname
    handleSymbol t = fallThrough t
    --
    -- ndarray operators
    --
    handleNDArray (ParamDescItem "NDArray-or-Symbol")   = symbol $ tyCon $ unQual $ name "NDArrayHandle"
    handleNDArray (ParamDescItem "NDArray-or-Symbol[]") = array $ tyList $ tyCon $ unQual $ name "NDArrayHandle"
    handleNDArray (ParamDescItem "Symbol or Symbol[]")  = array $ tyList $ tyCon $ unQual $ name "NDArrayHandle"
    handleNDArray (ParamDescItem "NDArray")             = symbol $ tyCon $ unQual $ name "NDArrayHandle"
    handleNDArray (ParamDescItem "Symbol")              = fail $ printf "SymbolHandle arg: %s" symname
    handleNDArray (ParamDescItem "Symbol[]")            = fail $ printf "SymbolHandle[] arg: %s" symname
    handleNDArray t = fallThrough t

    fallThrough t = fail $ printf "Unknown type: arg %s(%s)." symname desc

unQual = UnQual ()
unkindedVar = UnkindedVar ()

tyCon = TyCon ()
tyVarSymbol = TyVar () . Symbol ()
tyVarIdent = TyVar () . Ident ()
tyApp = TyApp ()
tyFun = TyFun ()
tySig names types = TypeSig () names types
tyList = TyList ()
tyVar = TyVar ()

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