
# decomposier_name_to_class = {
#     # 'BayesianDecomposition': BayesianDecomposition,
#     'NFDecomposition': NFDecomposition,
#     "TruncateDecomposition": TruncateDecomposition,
    
#     'NMFDecomposition': NMFDecomposition,
#     'PCADecomposition': PCADecomposition,
#     'SVDDecomposition': SVDDecomposition,
#     'ICADecomposition': ICADecomposition,
#     'FactorAnalysisDecomposition': FactorAnalysisDecomposition,
#     'DictionaryLearningDecomposition': DictionaryLearningDecomposition,
# }

# python eval_synthesizer.py --dataset_name "covtype" --synthesizer_name "CTGANSynthesizer" --num_train_samples 1000 --num_test_samples 1000 --num_train_epochs 10 --seed 42 --decomposer_name "PCADecomposition" --default_n_components 24
python eval_synthesizer.py --dataset_name "covtype" --synthesizer_name "CTGANSynthesizer" --num_train_samples 1000 --num_test_samples 1000 --num_train_epochs 10 --seed 42 --decomposer_name "SVDDecomposition"
python eval_synthesizer.py --dataset_name "adult" --synthesizer_name "CTGANSynthesizer" --num_train_samples 1000 --num_test_samples 1000 --num_train_epochs 10 --seed 42 --decomposer_name "FactorAnalysisDecomposition"
python eval_synthesizer.py --dataset_name "adult" --synthesizer_name "CTGANSynthesizer" --num_train_samples 1000 --num_test_samples 1000 --num_train_epochs 10 --seed 42 --decomposer_name "DictionaryLearningDecomposition"
python eval_synthesizer.py --dataset_name "adult" --synthesizer_name "CTGANSynthesizer" --num_train_samples 1000 --num_test_samples 1000 --num_train_epochs 10 --seed 42 --decomposer_name "ICADecomposition"
python eval_synthesizer.py --dataset_name "adult" --synthesizer_name "CTGANSynthesizer" --num_train_samples 1000 --num_test_samples 1000 --num_train_epochs 10 --seed 42 --decomposer_name "NMFDecomposition"
python eval_synthesizer.py --dataset_name "covtype" --synthesizer_name "CTGANSynthesizer" --num_train_samples 1000 --num_test_samples 1000 --num_train_epochs 10 --seed 42 --decomposer_name "NFDecomposition"
python eval_synthesizer.py --dataset_name "adult" --synthesizer_name "CTGANSynthesizer" --num_train_samples 1000 --num_test_samples 1000 --num_train_epochs 10 --seed 42 --decomposer_name "TruncateDecomposition"


python eval_synthesizer.py --dataset_name "adult" --synthesizer_name "CTGANSynthesizer" --num_train_samples 1000 --num_test_samples 1000 --num_train_epochs 10 --seed 42

python eval_synthesizer.py --dataset_name "covtype" --synthesizer_name "TVAESynthesizer" --num_train_samples 1000 --num_test_samples 1000 --num_train_epochs 1 --seed 42
