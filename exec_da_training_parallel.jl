using Distributed

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
    include("functions_spo.jl")
    include("functions_subgradient.jl")

end



function save_outcome(dict_outcome, dir)
    path = joinpath(dir, "outcome.csv")

    keys_outcome = string.(keys(dict_outcome))
    values_outcome = values(dict_outcome)

    CSV.write(path, Dict(key => value for (key, value) in zip(keys_outcome, values_outcome)))
end

function save_outcome_2(dict_outcome, dir)
    path = joinpath(dir, "outcome.csv")

    # Sort the keys of the dictionary
    sorted_keys = sort(string.(keys(dict_outcome)))

    # Create an empty DataFrame
    df = DataFrame()

    # Add columns to the DataFrame in sorted order
    for key in sorted_keys
        df[!, Symbol(key)] = dict_outcome[key]
    end

    # Write the DataFrame to a CSV file
    CSV.write(path, df)
    println("Saved")
end

function save_outcome_par(list_outcome_dicts,dir)
    
    for outcome_dict in list_outcome_dicts
        list_best = outcome_dict["net_best"]
        list_opti = outcome_dict["net_opti"]
        hp_config = outcome_dict["a_config"]
        dict_evols = outcome_dict["dict_evols"]
        list_fc = outcome_dict["list_fc_lists"]
        @show(list_fc[end])

        # for i in 1:length(list_best)
        #     store_code_best = "$(dir)config_$(hp_config)_best_$(i).npz"
        #     store_code_opti = "$(dir)config_$(hp_config)_opti_$(i).npz"

        #     npzwrite(store_code_best, list_best[i])
        #     npzwrite(store_code_opti,list_opti[i])
        # end

        save("$(dir)config_$(hp_config)_best_net.jld2","best_net", list_best)
        save("$(dir)config_$(hp_config)_opti_net.jld2","opti_net", list_opti)
        save("$(dir)config_$(hp_config)_train_evols.jld2","train_evols", dict_evols)
        save("$(dir)config_$(hp_config)_fc_evol.jld2","fc_evol",list_fc)       
    end

    dict_lists = convert_list_dict(list_outcome_dicts)

    pop!(dict_lists,"dict_evols")
    pop!(dict_lists,"list_fc_lists")
    pop!(dict_lists,"net_best")
    pop!(dict_lists,"net_opti")


    save_outcome_2(dict_lists,dir)

end

function convert_list_dict(list_dicts)
    #Function that converts list of dicts to dict of lists
    
    #Create empty dict with correct keys
    dict_lists = Dict()

    for key in keys(list_dicts[1])
        dict_lists[key] = []
    end

    #iteratively push values to correct list
    for config in 1:length(list_dicts)
        for key in keys(list_dicts[1])
            push!(dict_lists[key],list_dicts[config][key])
        end
    end

    return dict_lists
end

function get_loc_fc(data_dict,training_dict,machine)
    
    dirname = nothing
    act = training_dict["activation_function"]
    if machine == "local"
        dirname = "C:/Users/u0137781/OneDrive - KU Leuven/da_scheduling/Data/pretrained_fc/"
    elseif machine == "vsc"
        dirname = "../input_data/pretrained_fc/"
    end

    scaled_string = "unscaled"
    if data_dict["scale_price"]
        scaled_string = "scaled"
    end

    features_string = nothing
    if data_dict["feat_cols"] == ["weekday", "NL+FR","GEN_FC","y_hat"]
        features_string = "wd_nlfr_genfc_yhat"
    end

    filename = "$(dirname)model_$(act)_$(features_string)_$(scaled_string).bson"

    return filename

end














#User decisions

#run location, parallel or seq, make dir for saving results
machine = "local" #'local' or 'vsc'
par = false
makedir = false

#Store code
store_code = "20230925_subgradient_nonlinear_test4"

#HP combinations
dict_hps = Dict(
    "reg" => [0],#[0,0.001,0.1,10],
    "batch_size" => [8],
    "perturbation" => [0.1],#[0,0.5,0.1,0.02],
    "restrict_forecaster_ts" => false,#[true,false],
    "lr" => [30],
    "warm_start" => [false]
    )

#Determines the data sets
train_share = 1
days_train = floor(Int,1/train_share)
last_ex_test = 4 #59
repitition = 1

factor_size_ESS = 1
model_type = "LR"
la = 24

training_dict = Dict(
    "train_type" => "SG", #"IP" (Interior Point), "SG" (SubGradient) or "SL" (Simplex)
    "model_type" => "LR",
    "type_train_labels" => "price_schedule",
    "decomp" => "none",
    "train_mode" => "nonlinear", #"linear" or "nonlinear"
    "activation_function" => "softplus", #"relu" or "softplus" or "sigmoid"
    "pretrained_fc" => 1.0,
    "mu_update" => "auto", #"auto", "manual_d" (dynamic) or "manual_s" (static)
    "SG_epochs" => 200,
    "SG_patience" => 25,
    "SG_clip_base" => 1.0 #Value of gradient clipping, which is adjusted to the amount of updates and learning rate
)

OP_params_dict = Dict{Any,Any}(
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

loc_data = nothing
if machine == "local"
    loc_data = "C:/Users/u0137781/OneDrive - KU Leuven/da_scheduling/Data/processed_data/SPO_DA/"
elseif machine == "vsc"
    loc_data = "../input_data/processed_data/SPO_DA/"
end

data_dict = Dict(
    "loc_data" => loc_data,
    "feat_cols" => ["weekday", "NL+FR","GEN_FC","y_hat"],
    "col_label_price" => "y",
    "col_label_fc_price" => "y_hat",
    "lookahead" => la,
    "days_train" => days_train,
    "last_ex_test" => last_ex_test,
    "train_share" => train_share,
    "scale_mode" => "stand",
    "scale_base" => "y_hat",
    "scale_price" => true, #scale ground truth by dividing it with the stdev of the forecasted prices 
    "cols_no_centering" => ["y_hat"],
    "val_split_mode" => "alt_test" #'separate' for the validation set right before test set or 'alernating' for train/val examples alternating or "alt_test" for val_test examples alternating
)

OP_params_dict["pos_fc"] = findfirst(x -> x == "y_hat", data_dict["feat_cols"])
OP_params_dict["n_diff_features"] = length(data_dict["feat_cols"])





### Load pretrained forecaster
if training_dict["train_mode"] == "nonlinear"

    loc_fc = get_loc_fc(data_dict,training_dict,machine)
    println(loc_fc)
    # if machine == "local"
    #     loc_fc = "C:/Users/u0137781/OneDrive - KU Leuven/da_scheduling/Data/pretrained_fc/model_$(training_dict["activation_function"])_wd_nlfr_genfc_yhat_unscaled.bson"
    # elseif machine == "vsc"
    #     loc_fc = "../input_data/pretrained_fc/model_$(training_dict["activation_function"])_wd_nlfr_genfc_yhat_unscaled.bson"
    # end

    model = load(loc_fc)[:model_extended]

    pretrained_fc = [model[1].weight,model[1].bias,model[2].weight,model[2].bias]
    training_dict["pretrained_fc"] = pretrained_fc
    training_dict["pretrained_model"] = model

end



#Define datasets

features_train, features_val, features_test, price_train, price_val, price_test, _ , _ = preprocess_data(data_dict)

labels_train,labels_val,labels_test = preprocess_labels(training_dict["type_train_labels"],price_train,price_val,price_test,OP_params_dict,data_dict)

list_arrays = [features_train, labels_train, features_val, labels_val,features_test,labels_test]


#Actual training
list_input_dicts = []

if machine == "local"
    dir = "C:/Users/u0137781/OneDrive - KU Leuven/da_scheduling/Julia scripts/trained_models/$model_type/LA$(OP_params_dict["lookahead"])/SPO/$(store_code)/"
elseif machine == "vsc"
    dir = "../output/$(store_code)/"
end

if makedir
    mkdir(dir)
end



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

if par
    list_outcome_dicts = pmap(hp_tuning_spo_par,list_input_dicts)
else
    list_outcome_dicts = []
    for dict in list_input_dicts
        outcome_dict = hp_tuning_spo_par(dict)
        push!(list_outcome_dicts,outcome_dict)
    end
end



save_outcome_par(list_outcome_dicts,dir)




# x = list_outcome_dicts[1]["dict_evols"]["list_mu"]

# y = [x[i]/x[i-1] for i in 2:length(x)]

