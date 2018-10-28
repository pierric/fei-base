module Main where

import Options.Applicative
import Data.Semigroup ((<>))
import Language.Haskell.Exts
import qualified Data.Text as T
import System.Log.Logger
import Control.Monad
import Data.Either
import Data.Char (toLower, isUpper, isSpace, isAlphaNum)
import Text.Printf (printf)
import Text.ParserCombinators.ReadP

import MXNet.Base.Raw

_module_ = "Main"

data Arguments = Arguments {
    output_dir :: FilePath
}

args_spec = Arguments 
         <$> strOption (long "output" <> short 'o' <> metavar "OUTPUT-DIR")

main = do 
    args <- execParser opts

    ops  <- mxSymbolListAtomicSymbolCreators
    funs <- concat <$> mapM genSymOp ops

    -- putStrLn $ prettyPrint (mod funs)
    prettyPrint (mod funs) `seq` return ()
  where
    opts = info (args_spec <**> helper) (fullDesc <> progDesc "Generate MXNet operators")
    mod = Module () (Just $ ModuleHead () (ModuleName () "MXNet.Base.Symbol") Nothing Nothing) [] []

genSymOp :: AtomicSymbolCreator -> IO [Decl ()]
genSymOp sc = do
    (symname, desc, argname, argtype, argdesc, key_var_num_args, rettyp) <- mxSymbolGetAtomicSymbolInfo sc
    if symname `elem` ["_Native", "_NDArray"] then
        return []
    else do
        symname_ <- normalizeName symname
        let (errs, args) = partitionEithers $ zipWith resolveHaskellType argname argtype
        forM errs $ \(name, msg) -> 
            errorM _module_ (printf "Function: %s (%s, %s), %s" symname_ (show $ zip argname argtype) key_var_num_args msg)

        let paramList = map (\(name, typ) -> tyPromotedTuple [tyPromotedStr name, typ]) args 
            paramListInst = TypeInsDecl () 
                                (tyApp (tyCon $ unQual $ name "ParameterList") (tyPromotedStr symname_))
                                (tyPromotedList paramList)
            cxfullfill = appA (name "Fullfilled") [tyPromotedStr symname_, tyVarIdent "args"]

            typ = tyForall [unkindedVar (name "args")]
                    (cxSingle cxfullfill)
                    (tyFun 
                      (tyCon $ unQual $ name "String")
                      (tyFun 
                        (tyApp (tyApp (tyCon $ unQual $ name "HMap") (tyPromotedStr symname_)) (tyVarIdent "args")) 
                        (tyApp (tyCon $ unQual $ name "IO") (tyVarIdent "SymbolHandle"))))
            sig = tySig [name symname_] typ

        let dumps = function "dumps"
            unzip = function "unzip"
            keys = name "keys"
            vals = name "vals"
            fun = sfun (name symname_) [name "name", name "args"] (UnGuardedRhs () body) Nothing
            body =  letE [
                      patBind (pTuple [pvar keys, pvar vals]) (app unzip $ app dumps (var $ name "args"))
                    ] (doE [
                      genStmt (pvar $ name "op") $ function "NNGetOpHandle"
                        `app` strE symname
                    , genStmt (pvar $ name "sym") $ function "mxSymbolCreateAtomicSymbol" 
                        `app` (var $ name "op") 
                        `app` (var $ name "keys")
                        `app` (var $ name "vals")
                    , qualStmt $ function "mxSymbolCompose"
                        `app` (var $ name "sym")
                        `app` (var $ name "name")
                        `app` eList
                    , qualStmt $ function "return" `app` (var $ name "sym")
                    ])
        return [paramListInst, sig, fun]

normalizeName :: String -> IO String
normalizeName name@(c:cs) 
    | isUpper c = return $ toLower c:cs
    | otherwise = return $ name

data ParamDesc = ParamDescItem String | ParamDescList Bool [String] deriving (Eq, Show)

resolveHaskellType :: String -> String -> Either (String, String) (String, Type ())
resolveHaskellType symname desc = do
    let fields = runP desc
        optional = if ParamDescItem "optional" `elem` fields then True else False
        makeArgData = tyApp $ tyCon $ unQual $ name $ if optional then "AttrOpt" else "AttrReq"
        succ hstyp = Right (symname, hstyp)
        fail msg   = Left (symname, msg)    
    case head fields of 
        ParamDescItem "Shape(tuple)"        -> succ $ makeArgData $ tyList $ tyCon $ unQual $ name "Int"
        ParamDescItem "int"                 -> succ $ makeArgData $ tyCon $ unQual $ name "Int"
        ParamDescItem "int (non-negative)"  -> succ $ makeArgData $ tyCon $ unQual $ name "Int"
        ParamDescItem "long (non-negative)" -> succ $ makeArgData $ tyCon $ unQual $ name "Int"
        ParamDescItem "boolean"             -> succ $ makeArgData $ tyCon $ unQual $ name "Bool"
        ParamDescItem "float"               -> succ $ makeArgData $ tyCon $ unQual $ name "Float"
        ParamDescItem "double"              -> succ $ makeArgData $ tyCon $ unQual $ name "Double"
        ParamDescItem "float32"             -> succ $ makeArgData $ tyCon $ unQual $ name "Float"
        -- real_t (from mshadow) is by default float.
        ParamDescItem "real_t"              -> succ $ makeArgData $ tyCon $ unQual $ name "Float"
        ParamDescItem "string"              -> succ $ makeArgData $ tyCon $ unQual $ name "String"
        ParamDescItem "int or None"         -> succ $ makeArgData $ tyApp (tyCon $ unQual $ name "Maybe") (tyCon $ unQual $ name "Int")
        ParamDescItem "double or None"      -> succ $ makeArgData $ tyApp (tyCon $ unQual $ name "Maybe") (tyCon $ unQual $ name "Double")
        ParamDescItem "tuple of <float>"    -> succ $ makeArgData $ tyList $ tyCon $ unQual $ name "Float"
        ParamDescItem "tuple of <double>"   -> succ $ makeArgData $ tyList $ tyCon $ unQual $ name "Double"
        ParamDescList hasnone vs -> do
            let vsprom = map tyPromotedStr vs
                typ1 = tyApp (tyCon $ unQual $ name "EnumType") (tyPromotedList vsprom)
                typ2 = tyApp (tyCon $ unQual $ name "Maybe") typ1
            succ $ makeArgData $ if hasnone then typ2 else typ1

        ParamDescItem "Symbol"              -> succ $ makeArgData $ tyCon $ unQual $ name "SymbolHandle"
        ParamDescItem "NDArray"             -> succ $ makeArgData $ tyCon $ unQual $ name "NDArrayHandle"
        ParamDescItem "NDArray-or-Symbol"   -> succ $ makeArgData $ tyCon $ unQual $ name "SymbolHandle"
        -- operator can have only one argument of the type symbol or ndarray array
        -- and not having any other argument of symbol or ndarray
        -- besides the operator's info must definitely have a key_var_num_args, which
        -- indicates an (might be additional) argument which should be passed to mxSymbolCreateAtomicSymbol.
        ParamDescItem "Symbol[]"            -> succ $ makeArgData $ tyList $ tyCon $ unQual $ name "SymbolHandle"
        ParamDescItem "NDArray-or-Symbol[]" -> succ $ makeArgData $ tyList $ tyCon $ unQual $ name "SymbolHandle"
        ParamDescItem "Symbol or Symbol[]"  -> succ $ makeArgData $ tyList $ tyCon $ unQual $ name "SymbolHandle"

        t -> fail $ printf "Unknown type: arg %s(%s)." symname desc
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