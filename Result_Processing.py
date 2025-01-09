import pandas as pd
import numpy as np


validation_overall_accuracies = []

validation_weighted_accuracies = []
validation_weighted_f1s = []
validation_weighted_precisions = []

validation_settings = []
validation_sizes = []
validation_experiments = []

for e_i, experiment in enumerate(["Spatial","Spatio-Temporal"]): # Perform Validation Result Processing for Each Subset 
    for size in [5,10,25,50]:
        for setting in [1,2,3]:
            current_path = f"Transformed_Selected_Sen4AgriNet_Dataset_Results/Transformation_Setting_{setting}/{size}%/Experiment_{e_i+2}/Fold_1/Validation_Results/"
            counts_path = f"Transformed_Selected_Sen4AgriNet_Dataset_Info/Transformation_Setting_{setting}/{size}%/Experiment_{e_i+2}/Fold_1/Class_Counts_Validation_Set.txt"

            with open(current_path + "Overall_Accuracy.txt", "r") as aAcc_file:
                aAcc = aAcc_file.read()
            
            with open(counts_path, "r") as counts_file:
                counts = [float(c) for c in counts_file]

            if aAcc != '':
                validation_overall_accuracies.append(float(aAcc))

                with open(current_path + "Accuracies.txt", "r") as Accs_file: # Calculate Weighted Accuracy according to Accuracy of each class
                    Accs = [float(Acc) for Acc in Accs_file]
                    wAccs = sum([(Accs[i]*counts[i])/sum(counts) for i in range(len(Accs))])

                    validation_weighted_accuracies.append(np.mean(np.array(wAccs)))
                
                with open(current_path + "F1_Scores.txt", "r") as F1s_file: # Calculate Weighted F1 according to F1 of each class
                    F1s = [float(F1) for F1 in F1s_file]
                    wF1s = sum([(F1s[i]*counts[i])/sum(counts) for i in range(len(F1s))])
                    
                    validation_weighted_f1s.append(np.mean(np.array(wF1s)))
                
                with open(current_path + "Precisions.txt", "r") as Precs_file: # Calculate Weighted Precision according to Precision of each class  
                    Precs = [float(Prec) for Prec in Precs_file]
                    wPrecs = sum([(Precs[i]*counts[i])/sum(counts) for i in range(len(Precs))])
                    
                    validation_weighted_precisions.append(np.mean(np.array(wPrecs)))
                
                validation_settings.append(setting)
                validation_sizes.append(str(size)+'\%')
                validation_experiments.append(experiment)

val_results_df = pd.DataFrame({"Experiment": validation_experiments,
                               "Size": validation_sizes,
                               "Setting": validation_settings,
                               "W. Acc.": validation_weighted_accuracies,
                               "W. F1": validation_weighted_f1s,
                               "W. Prec.": validation_weighted_precisions,
                                })
val_results_df_latex = val_results_df.to_latex(index=False)

val_results_df.to_csv("Processed_Results/Processed_Validation_Results.csv")
with open("Processed_Results/Processed_Validation_Results.txt", "w") as file:
    file.write(val_results_df_latex)
print(val_results_df)

test_overall_accuracies = []

test_weighted_accuracies = []
test_weighted_f1s = []
test_weighted_precisions = []

test_settings = []
test_sizes = []
test_experiments = []

for e_i, experiment in enumerate(["Spatial","Spatio-Temporal"]): # Perform Test Result Processing for Each Subset 
    for size in [5,10,25,50]:
        for setting in [1,2,3]:
            current_path = f"Transformed_Selected_Sen4AgriNet_Dataset_Results/Transformation_Setting_{setting}/{size}%/Experiment_{e_i+2}/Fold_1/Test_Results/"
            counts_path = f"Transformed_Selected_Sen4AgriNet_Dataset_Info/Transformation_Setting_{setting}/{size}%/Experiment_{e_i+2}/Fold_1/Class_Counts_Test_Set.txt"

            with open(current_path + "Overall_Accuracy.txt", "r") as aAcc_file:
                aAcc = aAcc_file.read()
            
            with open(counts_path, "r") as counts_file:
                counts = [float(c) for c in counts_file]

            if aAcc != '':
                test_overall_accuracies.append(float(aAcc))

                with open(current_path + "Accuracies.txt", "r") as Accs_file: # Calculate Weighted Accuracy according to Accuracy of each class
                    Accs = [float(Acc) for Acc in Accs_file]
                    wAccs = sum([(Accs[i]*counts[i])/sum(counts) for i in range(len(Accs))])

                    test_weighted_accuracies.append(np.mean(np.array(wAccs)))
                
                with open(current_path + "F1_Scores.txt", "r") as F1s_file: # Calculate Weighted F1 according to F1 of each class   
                    F1s = [float(F1) for F1 in F1s_file]
                    wF1s = sum([(F1s[i]*counts[i])/sum(counts) for i in range(len(F1s))])
                    
                    test_weighted_f1s.append(np.mean(np.array(wF1s)))
                
                with open(current_path + "Precisions.txt", "r") as Precs_file: # Calculate Weighted Precision according to Precision of each class     
                    Precs = [float(Prec) for Prec in Precs_file]
                    wPrecs = sum([(Precs[i]*counts[i])/sum(counts) for i in range(len(Precs))])
                    
                    test_weighted_precisions.append(np.mean(np.array(wPrecs)))
                
                test_settings.append(setting)
                test_sizes.append(str(size)+'\%')
                test_experiments.append(experiment)

test_results_df = pd.DataFrame({"Experiment": test_experiments,
                                "Size": test_sizes,
                                "Setting": test_settings,
                                "W. Acc.": test_weighted_accuracies,
                                "W. F1": test_weighted_f1s,
                                "W. Prec.": test_weighted_precisions,
                                })
test_results_df_latex = test_results_df.to_latex(index=False)

test_results_df.to_csv("Processed_Results/Processed_Test_Results.csv")
with open("Processed_Results/Processed_Test_Results.txt", "w") as file:
    file.write(test_results_df_latex)
print(test_results_df)
