using Distributed

#add more procs for parallel implementation
while nprocs() <1
    addprocs(1)
end

@everywhere begin    

    using Flux
    using Flux: Chain, Dense, softplus, relu
    using Random
    using CSV
    using DataFrames
    using Dates
    using Statistics
    using LinearAlgebra
    #using JLD
    using JLD2
    using DelimitedFiles
    using NPZ
    using JuMP
    using Gurobi
    using Ipopt
    using BSON: @save, load
    using SparseArrays


    include("functions_support.jl")
    include("../training/functions_spo.jl")
    include("../training/functions_subgradient.jl")

end





##### USER DECISIONS #####

#Definition of the model
reforecast_type = "Sp_IP" # "Sp_SG", "Sp_IP", "Sp_IPs" or "Sp_IPd"
nn_type = "linear" # "linear" or "softplus"
warm_start = false

#Definition of hyperparamters
dict_hps = Dict(
    #choice of hyperaparemeters to be tuned. The training procedure will exploit all possible combinations (grid search)
    "reg" => [0], #Regularizer: [0,0.001,0.1,10] for "Sp_IP", [0,0.0001,0.01,1] for "Sp_SG"
    "batch_size" => [64],  #[64] for "Sp_IP", [8,64] for "Sp_SG"
    "perturbation" => [0.1], #Allowed perturbation of re-forecaster compared to initial FC: [0,0.5,0.1,0.02] for "Sp_IP", [0] for "Sp_SG" (0 meaning no constraint)
    "restrict_forecaster_ts" => [true], #whether or not we restrict the re-forecaster to make predictions based only on features of current timestep [true,false] for "Sp_IP", [false] for "Sp_SG"
    "lr" => [0], #Start point of learning rate for subgradient method: "[0] for "Sp_IP", [0.001,0.01,0.1,1] for "Sp_SG"
    )

#Store code; A folder with this name will be created in ../training/train/outcome/ containing all the information of the training procedure
store_code = "test"






###### DEFINITION OF FIXED SETTINGS #####    

#run location, parallel or seq, make dir for saving results
machine = "local"
par = false #true for parallel training
makedir = false

#Paremeters determining the dataset
train_share = 1
days_train = floor(Int,64/train_share)
last_ex_test = 59 #59
repitition = 1

factor_size_ESS = 1
model_type = "LR"
la = 24

training_dict = Dict(
    #Dict containing info on how the forecaster is trained
    "model_type" => "LR",
    "type_train_labels" => "price_schedule",
    "decomp" => "none",
    "activation_function" => "softplus", #"relu" or "softplus" or "sigmoid"
    "pretrained_fc" => 1.0,
    "SG_epochs" => 200,
    "SG_patience" => 25,
    "SG_clip_base" => 1.0 #Value of gradient clipping, which is adjusted to the amount of updates and learning rate
)

OP_params_dict = Dict{Any,Any}(
    #Dict containing info of optimization program
    "max_charge" => 0.01 * factor_size_ESS,
    "max_discharge" => 0.01 * factor_size_ESS,
    "eff_d" => 0.95,
    "eff_c" => 0.95,
    "max_soc" => 0.04 * factor_size_ESS,
    "min_soc" => 0,
    "soc_0" => 0.02 * factor_size_ESS,
    "ts_len" => 1,
    "lookahead" => la,
    "soc_update" => true,
    "cyclic_bc" => true,
)

data_dict = Dict(
    #Dict containing all info required to retrieve and handle data
    "feat_cols" => ["weekday", "NL+FR","GEN_FC","y_hat"],
    "col_label_price" => "y",
    "col_label_fc_price" => "y_hat",
    "lookahead" => la,
    "days_train" => days_train,
    "last_ex_test" => last_ex_test,
    "train_share" => train_share,
    "scale_mode" => "stand", #standardize
    "scale_base" => "y_hat",
    "scale_price" => true, #scale ground truth by dividing it with the stdev of the forecasted prices 
    "cols_no_centering" => ["y_hat"], #inlcude columns that should not be adjusted for their mean when scaling. We apply this for price data to avoid undesired effects of negative prices
    "val_split_mode" => "alt_test" #'separate' for the validation set right before test set or 'alernating' for train/val examples alternating or "alt_test" for val_test examples alternating
)

loc_data = nothing
if machine == "local"
    loc_data = "./data/processed_data/SPO_DA/"
elseif machine == "vsc"
    loc_data = "../input_data/processed_data/SPO_DA/"
end


#Extend dicts
train_type,train_mode,mu_update = get_type_mode_mu(reforecast_type,nn_type)
training_dict["train_type"] = train_type
training_dict["train_mode"] = train_mode
training_dict["mu_update"] = mu_update
dict_hps["warm_start"] = [warm_start]
data_dict["loc_data"] = loc_data
OP_params_dict["pos_fc"] = findfirst(x -> x == "y_hat", data_dict["feat_cols"])
OP_params_dict["n_diff_features"] = length(data_dict["feat_cols"])


# Load pretrained forecaster
if training_dict["train_mode"] == "nonlinear"

    loc_fc = get_loc_fc(data_dict,training_dict,machine)

    model = load(loc_fc)[:model_extended]

    pretrained_fc = [model[1].weight,model[1].bias,model[2].weight,model[2].bias]
    training_dict["pretrained_fc"] = pretrained_fc
    training_dict["pretrained_model"] = model

end





##### LOADING DATA AND TRAINING #####
#Define datasets

features_train, features_val, features_test, price_train, price_val, price_test, _ , _ = preprocess_data(data_dict)

labels_train,labels_val,labels_test = preprocess_labels(training_dict["type_train_labels"],price_train,price_val,price_test,OP_params_dict,data_dict)

list_arrays = [features_train, labels_train, features_val, labels_val,features_test,labels_test]


#Create folder to store training information
if machine == "local"
    dir = "./training/train_output/$(store_code)/"
elseif machine == "vsc"
    dir = "../output/$(store_code)/"
end

if makedir
    mkdir(dir)
end

#Create list to store all input information for the different HP combinations
list_input_dicts = []
input_dict = Dict(
    "params_dict" => OP_params_dict,
    "training_dict" => training_dict,
    "model_type" => model_type,
    "list_tensors" => list_arrays,
    "store_code" => store_code,
    "dir" => dir
)

hp_config = 1
for reg in dict_hps["reg"]
    for batch_size in dict_hps["batch_size"]
        for pert in dict_hps["perturbation"]
            for restrict in dict_hps["restrict_forecaster_ts"]
                for lr in dict_hps["lr"]
                    for ws in dict_hps["warm_start"]
                        for _ in 1:repitition
                        
                            global hp_config

                            new_input_dict = deepcopy(input_dict)
                            
                            new_input_dict["reg"] = reg
                            new_input_dict["batch_size"] = batch_size
                            new_input_dict["pert"] = pert
                            new_input_dict["restrict"] = restrict
                            new_input_dict["lr"] = lr
                            new_input_dict["warm_start"] = ws

                            new_input_dict["hp_config"] = hp_config
                            hp_config+=1
                            push!(list_input_dicts,new_input_dict)
                        end
                    end
                end                
            end
        end
    end
end

#Actual training, calling hp_tuning_spo_par for hyperparameter tuning
if par
    list_outcome_dicts = pmap(hp_tuning_spo_par,list_input_dicts)
else
    list_outcome_dicts = []
    for dict in list_input_dicts
        outcome_dict = hp_tuning_spo_par(dict)
        push!(list_outcome_dicts,outcome_dict)
    end
end


#Save outcome
save_outcome_par(list_outcome_dicts,dir)
