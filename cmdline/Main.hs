module Main where

import Options.Applicative
import Data.Semigroup ((<>))
import Language.Haskell.Exts
import qualified Data.Text as T
import System.Log.Logger
import Control.Monad
import qualified Control.Monad.RWS as M
import qualified Control.Monad.State as M
import Data.Either
import Data.Bifunctor
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

    putStrLn $ prettyPrint (mod funs)
    -- prettyPrint (mod funs) `seq` return ()
  where
    opts = info (args_spec <**> helper) (fullDesc <> progDesc "Generate MXNet operators")
    mod = Module () (Just $ ModuleHead () (ModuleName () "MXNet.Base.Symbol") Nothing Nothing) [] []

genSymOp :: AtomicSymbolCreator -> IO [Decl ()]
genSymOp sc = do
    (name, desc, argname, argtype, argdesc, key_var_num_args, rettyp) <- mxSymbolGetAtomicSymbolInfo sc
    if name `elem` ["_Native", "_NDArray"] then
        return []
    else do
        name_ <- normalizeName name
        let (errs, args) = partitionEithers $ zipWith resolveHaskellType argname argtype
        forM errs $ \(name, msg) -> 
            errorM _module_ (printf "Function: %s (-> %s), %s" name_ rettyp msg)

        let paramList = map (\(name, typ) -> tyPromotedTuple [tyPromotedStr name, typ]) args 
            paramListInst = TypeInsDecl () 
                                (tyApp (tyCon $ unQual $ ident "ParameterList") (tyPromotedStr name_))
                                (tyPromotedList paramList)
            cxfullfill = appA (ident "Fullfilled") [tyPromotedStr name_, tyVarIdent "args"]

            typ = tyForall [unkindedVar (symbol "args")] 
                    (cxSingle cxfullfill)
                    (tyFun (tyApp (tyApp (tyCon $ unQual $ ident "HMap") (tyPromotedStr name_)) (tyVarIdent "args")) (tyApp (tyCon $ unQual $ ident "IO") (tyVarIdent "SymbolHandle")))
            sig = tySig [ident name_] typ
        return [paramListInst, sig]

normalizeName :: String -> IO String
normalizeName name@(c:cs) 
    | isUpper c = return $ toLower c:cs
    | otherwise = return $ name

data ParamDesc = ParamDescItem String | ParamDescList Bool [String] deriving (Eq, Show)

resolveHaskellType :: String -> String -> Either (String, String) (String, Type ())
resolveHaskellType name desc = do
    let fields = runP desc
        optional = if ParamDescItem "optional" `elem` fields then True else False
        makeArgData = tyApp $ tyCon $ unQual $ ident $ if optional then "AttrOpt" else "AttrReq"
        succ hstyp = Right (name, hstyp)
        fail msg   = Left (name, msg)    
    case head fields of 
        ParamDescItem "NDArray-or-Symbol"   -> succ $ makeArgData $ tyCon $ unQual $ ident "SymbolHandle"
        ParamDescItem "Shape(tuple)"        -> succ $ makeArgData $ tyList $ tyCon $ unQual $ ident "Int"
        ParamDescItem "int"                 -> succ $ makeArgData $ tyCon $ unQual $ ident "Int"
        ParamDescItem "int (non-negative)"  -> succ $ makeArgData $ tyCon $ unQual $ ident "Int"
        ParamDescItem "long (non-negative)" -> succ $ makeArgData $ tyCon $ unQual $ ident "Int"
        ParamDescItem "boolean"             -> succ $ makeArgData $ tyCon $ unQual $ ident "Bool"
        ParamDescItem "float"               -> succ $ makeArgData $ tyCon $ unQual $ ident "Float"
        ParamDescItem "double"              -> succ $ makeArgData $ tyCon $ unQual $ ident "Float"
        ParamDescItem "float32"             -> succ $ makeArgData $ tyCon $ unQual $ ident "Float"
        ParamDescItem "real_t"              -> succ $ makeArgData $ tyCon $ unQual $ ident "Float"
        ParamDescItem "string"              -> succ $ makeArgData $ tyCon $ unQual $ ident "String"
        ParamDescItem "int or None"         -> succ $ makeArgData $ tyApp (tyCon $ unQual $ ident "Maybe") (tyCon $ unQual $ ident "Int")
        ParamDescItem "double or None"      -> succ $ makeArgData $ tyApp (tyCon $ unQual $ ident "Maybe") (tyCon $ unQual $ ident "Float")
        ParamDescItem "tuple of <float>"    -> succ $ makeArgData $ tyList $ tyCon $ unQual $ ident "Float"
        ParamDescItem "Symbol"              -> succ $ makeArgData $ tyCon $ unQual $ ident "SymbolHandle"
        ParamDescItem "NDArray"             -> succ $ makeArgData $ tyCon $ unQual $ ident "NDArrayHandle"
        ParamDescItem "Symbol[]"            -> succ $ makeArgData $ tyList $ tyCon $ unQual $ ident "SymbolHandle"
        ParamDescItem "NDArray-or-Symbol[]" -> succ $ makeArgData $ tyList $ tyCon $ unQual $ ident "SymbolHandle"
        ParamDescItem "Symbol or Symbol[]"  -> succ $ makeArgData $ tyList $ tyCon $ unQual $ ident "SymbolHandle"
        ParamDescList hasnone vs -> do
            let vsprom = map tyPromotedStr vs
                typ1 = tyApp (tyCon $ unQual $ ident "EnumType") (tyPromotedList vsprom)
                typ2 = tyApp (tyCon $ unQual $ ident "Maybe") typ1
            succ $ makeArgData $ if hasnone then typ2 else typ1
        t -> fail $ printf "Unknown type: arg %s(%s)." name desc
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

tyCon = TyCon ()
tyVarSymbol = TyVar () . Symbol ()
tyVarIdent = TyVar () . Ident ()
tyApp = TyApp ()
tyFun = TyFun ()
tySig names types = TypeSig () names types
tyList = TyList ()
tyVar = TyVar ()
ident = Ident ()
symbol = Symbol ()

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

unkindedVar = UnkindedVar ()

appA = AppA ()

unQual = UnQual ()