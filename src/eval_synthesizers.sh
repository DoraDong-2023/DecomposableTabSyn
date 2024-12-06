# Experiment 1: on covtype, original synthesizer and decompose-based synthesizers, Quality report; Table 
dataset_name="covtype"
num_test_samples=2000
num_train_samples=5000
num_train_epochs=5
seed=42
for synthesizer_name in "CTGANSynthesizer" "TVAESynthesizer" "CopulaGANSynthesizer" "GaussianCopulaSynthesizer" "Tabula"; do
    for decomposer_name in "no_decomposition" "PCADecomposition" "SVDDecomposition" "FactorAnalysisDecomposition" "DictionaryLearningDecomposition" "ICADecomposition" "NMFDecomposition" "NFDecomposition" "TruncateDecomposition"; do
        echo "Running experiment with dataset_name: $dataset_name, synthesizer_name: $synthesizer_name, num_train_samples: $num_train_samples, num_test_samples: $num_test_samples, num_train_epochs: $num_train_epochs, seed: $seed, decomposer_name: $decomposer_name"
        python eval_synthesizer.py --dataset_name $dataset_name --synthesizer_name $synthesizer_name --num_train_samples $num_train_samples --num_test_samples $num_test_samples --num_train_epochs $num_train_epochs --seed $seed --decomposer_name $decomposer_name
    done
done

# Experiment 2: on Covtype, Original synthesizer and decompose-based synthesizers performance trends across different number of training samples, fitting time, and number of epochs
# Figure; x-axis: number of training samples, y-axis: quality, fitting time
dataset_name="covtype"
num_test_samples=2000
num_train_epochs=5
seed=42
for num_train_samples in 1000 5000 10000 50000 100000; do
    for synthesizer_name in "TVAESynthesizer"; do
        for decomposer_name in "no_decomposition" "PCADecomposition" "NFDecomposition" "TruncateDecomposition"; do
            echo "Running experiment with dataset_name: $dataset_name, synthesizer_name: $synthesizer_name, num_train_samples: $num_train_samples, num_test_samples: $num_test_samples, num_train_epochs: $num_train_epochs, seed: $seed, decomposer_name: $decomposer_name"
            python eval_synthesizer.py --dataset_name $dataset_name --synthesizer_name $synthesizer_name --num_train_samples $num_train_samples --num_test_samples $num_test_samples --num_train_epochs $num_train_epochs --seed $seed --decomposer_name $decomposer_name
        done
    done
done

# Experiment 3: on Covtype, Original synthesizer and decompose-based synthesizers performance trends across different number of test samples, inference time, and number of epochs
# Figure; x-axis: number of test samples, y-axis: quality, inference time
dataset_name="covtype"
num_train_samples=5000
num_train_epochs=5
seed=42
for num_test_samples in 1000 2000 4000 8000 16000; do
    for synthesizer_name in "TVAESynthesizer"; do
        for decomposer_name in "no_decomposition" "PCADecomposition" "NFDecomposition" "TruncateDecomposition"; do
            echo "Running experiment with dataset_name: $dataset_name, synthesizer_name: $synthesizer_name, num_train_samples: $num_train_samples, num_test_samples: $num_test_samples, num_train_epochs: $num_train_epochs, seed: $seed, decomposer_name: $decomposer_name"
            python eval_synthesizer.py --dataset_name $dataset_name --synthesizer_name $synthesizer_name --num_train_samples $num_train_samples --num_test_samples $num_test_samples --num_train_epochs $num_train_epochs --seed $seed --decomposer_name $decomposer_name
        done
    done
done


# Experiment 4: on various datasets, Original synthesizer and decompose-based synthesizers performance trends across different number of columns
# Figure; x-axis: number of columns, y-axis: quality, fitting time, inference time
dataset_name="asia,adult,insurance,alarm,covtype,mnist12" # different number of columns; different data structures
num_train_epochs=5
synthesizer_name="TVAESynthesizer"
for decomposer_name in "no_decomposition" "PCADecomposition"; do
    for num_train_samples in 1000 5000 10000; do
        for num_test_samples in 1000 2000 4000; do
            echo "Running experiment with dataset_name: $dataset_name, synthesizer_name: $synthesizer_name, num_train_samples: $num_train_samples, num_test_samples: $num_test_samples, num_train_epochs: $num_train_epochs, seed: $seed, decomposer_name: $decomposer_name"
            python eval_synthesizer.py --dataset_name $dataset_name --synthesizer_name $synthesizer_name --num_train_samples $num_train_samples --num_test_samples $num_test_samples --num_train_epochs $num_train_epochs --seed $seed --decomposer_name $decomposer_name    
        done
    done
done


# Experiment 4 (Tabula): on various datasets, Original synthesizer and decompose-based synthesizers performance trends across different number of columns
# Figure; x-axis: number of columns, y-axis: quality, fitting time, inference time
dataset_name="asia,adult,insurance,alarm,covtype,mnist12" # different number of columns; different data structures
num_train_epochs=5
synthesizer_name="Tabula"
seed=42
for decomposer_name in "no_decomposition" "PCADecomposition"; do
    for num_train_samples in 1000 5000 10000; do
        for num_test_samples in 1000 2000 4000; do
            echo "Running experiment with dataset_name: $dataset_name, synthesizer_name: $synthesizer_name, num_train_samples: $num_train_samples, num_test_samples: $num_test_samples, num_train_epochs: $num_train_epochs, seed: $seed, decomposer_name: $decomposer_name"
            python eval_synthesizer.py --dataset_name $dataset_name --synthesizer_name $synthesizer_name --num_train_samples $num_train_samples --num_test_samples $num_test_samples --num_train_epochs $num_train_epochs --seed $seed --decomposer_name $decomposer_name    
        done
    done
done

# Experiment 5: on Covtype, adult and insurance, How does the performance of PCAdecomposition change with the number of components, quality, inference time, and fitting time
# Figure; x-axis: number of components, y-axis: quality, fitting time, inference time; no_decomposition as baseline
dataset_name="covtype" # 47 columns
num_train_epochs=5
num_train_samples=5000
num_test_samples=2000
synthesizer_name="CTGANSynthesizer"
for decomposer_name in "PCADecomposition"; do
    for default_n_components in 2 4 8 16 24 32; do
        echo "Running experiment with dataset_name: $dataset_name, synthesizer_name: $synthesizer_name, num_train_samples: $num_train_samples, num_test_samples: $num_test_samples, num_train_epochs: $num_train_epochs, seed: $seed, decomposer_name: $decomposer_name"
        python eval_synthesizer.py --dataset_name $dataset_name --synthesizer_name $synthesizer_name --num_train_samples $num_train_samples --num_test_samples $num_test_samples --num_train_epochs $num_train_epochs --seed $seed --decomposer_name $decomposer_name --default_n_components $default_n_components
    done
done
