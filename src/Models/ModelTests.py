import PredictFromAugmentedRealMixtures.ModelLoader as aug
import PredictFromPseudoMixtures.ModelLoader as pseudo
import PredictFishTypeFromSingle.ModelLoader as single

device, model, transform = pseudo.loadModel()