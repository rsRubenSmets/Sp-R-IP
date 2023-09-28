using Flux
using Random
using CSV
using DataFrames
using Dates
using Statistics
using LinearAlgebra
using JLD2
using Gurobi
using DelimitedFiles
using HDF5
using JuMP
using NPZ
using Plots
using DataStructures
include("../experiment/functions_support.jl")
include("functions_support_results.jl")


#### Obtaining results of Table 1 and Table 2  #####


#Preprocessing
list_dirs_cold = [
    "ID_Q_linear_cold",
    "ID_Q_softplus_cold",
    "Sp_SG_linear_cold",
    "Sp_SG_softplus_cold",
    "Sp_R_IP_linear_cold",
    "Sp_R_IP_softplus_cold",
    "Sp_R_IPs_linear_cold",
    "Sp_R_IPs_softplus_cold",
    "Sp_R_IPd_linear_cold",
    "Sp_R_IPd_softplus_cold",
]

list_dirs_warm = [
    "ID_Q_linear_warm",
    "ID_Q_softplus_warm",
    "Sp_SG_linear_warm",
    "Sp_SG_softplus_warm",
    "Sp_R_IP_linear_warm",
    "Sp_R_IP_softplus_warm",
    "Sp_R_IPs_linear_warm",
    "Sp_R_IPs_softplus_warm",
    "Sp_R_IPd_linear_warm",
    "Sp_R_IPd_softplus_warm",
]

list_dirs_reform = [ #Reformulation models (Sp-R) treated differently because for those, it's the validation regret for the optimized solution that counts for finding the best model
    "Sp_R_IP_linear_cold",
    "Sp_R_IP_softplus_cold",
    "Sp_R_IP_linear_warm",
    "Sp_R_IP_softplus_warm"
]


list_evols_cold,_ = read_evol_lists(list_dirs_cold,false)
list_evols_red_cold = reduce_evol_lists(list_evols_cold)
dfs_best_cold = get_dataframes_best(list_evols_red_cold)
df_best_sorted_cold,best_configs_cold = sort_df(dfs_best_cold)
list_outcomes_cold = get_outcomes(list_dirs_cold)

list_evols_warm,_ = read_evol_lists(list_dirs_warm,false)
list_evols_red_warm = reduce_evol_lists(list_evols_warm)
dfs_best_warm = get_dataframes_best(list_evols_red_warm)
df_best_sorted_warm,best_configs_warm = sort_df(dfs_best_warm)
list_outcomes_warm = get_outcomes(list_dirs_warm)

list_outcomes_reform = get_outcomes(list_dirs_reform)
df_best_sorted_reform, best_configs_reform = sort_df(list_outcomes_reform,"outcome")

profit_pf_val = get_profit(list_outcomes_cold,best_configs_cold,"PF", "val")
profit_pf_test = get_profit(list_outcomes_cold,best_configs_cold,"PF", "test")
profit_fc_val = get_profit(list_outcomes_cold,best_configs_cold,"fc", "val")
profit_fc_test = get_profit(list_outcomes_cold,best_configs_cold,"fc", "test")
regret_fc_val = profit_pf_val - profit_fc_val
regret_fc_test = profit_pf_test - profit_fc_test



#Values of abs regret in Table 1
test_regret_cold = get_properties_best(dfs_best_cold,best_configs_cold,"regret_test")
test_regret_warm = get_properties_best(dfs_best_warm,best_configs_warm,"regret_test")
test_regret_reform = calc_property_outcome(list_outcomes_reform, best_configs_reform, "regret", "test", true)

#Values of rel regret in Table 1
test_regret_impr_cold = [(test_regret_cold[i]-regret_fc_test[i])/regret_fc_test[i] for i in 1:length(test_regret_cold)]
test_regret_impr_warm = [(test_regret_warm[i]-regret_fc_test[i])/regret_fc_test[i] for i in 1:length(test_regret_warm)]
test_regret_impr_reform = [(test_regret_reform[i]-regret_fc_test[i])/regret_fc_test[i] for i in 1:length(test_regret_reform)]

#Values of train time in Table 1 and Table 2
train_times_cold = get_property_outcome_config(list_outcomes_cold,best_configs_cold,"b_train_time")
train_times_warm = get_property_outcome_config(list_outcomes_warm,best_configs_warm,"b_train_time")
train_times_reform = get_property_outcome_config(list_outcomes_reform,best_configs_reform,"b_train_time")


#Values of abs regret in Table 2
val_regret_cold = get_properties_best(dfs_best_cold,best_configs_cold,"regret_val")
val_regret_warm = get_properties_best(dfs_best_warm,best_configs_warm,"regret_val")
val_regret_reform = calc_property_outcome(list_outcomes_reform, best_configs_reform, "regret", "val", true)

#Values of rel regret in Table 2
val_regret_impr_cold = [(val_regret_cold[i]-regret_fc_val[i])/regret_fc_val[i] for i in 1:length(val_regret_cold)]
val_regret_impr_warm = [(val_regret_warm[i]-regret_fc_val[i])/regret_fc_val[i] for i in 1:length(val_regret_warm)]
val_regret_impr_reform = [(val_regret_reform[i]-regret_fc_val[i])/regret_fc_val[i] for i in 1:length(val_regret_reform)]
