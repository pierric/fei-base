module MXNet.Base.Symbol where

import qualified MXNet.Base.Raw as I

newtype Symbol a = Symbol { unSymbol :: I.SymbolHandle }

listArguments :: Symbol a -> IO [String]
listArguments (Symbol sym) = I.mxSymbolListArguments sym

listOutputs :: Symbol a -> IO [String]
listOutputs (Symbol sym) = I.mxSymbolListOutputs sym

listAuxiliaryStates :: Symbol a -> IO [String]
listAuxiliaryStates (Symbol sym) = I.mxSymbolListAuxiliaryStates sym