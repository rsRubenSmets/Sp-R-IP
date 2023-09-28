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



function read_evol_lists(list_dirs,fc_evol=false)

    list_evol_lists = []
    list_fc_lists = []
    for dir in list_dirs
        dir,suffix = locate_dir(dir)
        println("Dir: $(dir); suffix: $(suffix)")
        list_evols = []
        c = 1
        #label = nothing

        loc = "$(dir)/config_$(c)_train_evols.$(suffix)"
        loc_fc = "$(dir)/config_$(c)_fc_evol.jld2"
        println("loc: $(loc)")
        while isfile(loc)
            if suffix == "jld2"
                push!(list_evols,JLD2.load(loc)["train_evols"])
                if fc_evol
                    push!(list_fc_lists,JLD2.load(loc_fc)["fc_evol"])
                end               
            elseif suffix == "h5"
                data = Dict()
                h5open(loc, "r") do file
                    for key in keys(file)
                        data[key] = read(file[key])
                    end
                end
                push!(list_evols,data)
            end
            c+=1
            loc = "$(dir)/config_$(c)_train_evols.$(suffix)"
            loc_fc = "$(dir)/config_$(c)_fc_evol.jld2"
        end
        push!(list_evol_lists,list_evols)
    end
    return list_evol_lists,list_fc_lists
end

function locate_dir(dir)
    base_dir = "./training/train_output/"


    if occursin("ID_",dir) 
        dir = base_dir*dir
        return dir,"h5"
    elseif occursin("_IP",dir) || occursin("_SG",dir)
        dir = base_dir*dir
        return dir, "jld2"
    else
        error("Invalid dir name: $(dir)")
    end
end

function reduce_evol_lists(list_evol_lists)
    i=1
    list_red = []
    for list in list_evol_lists
        if haskey(list[1],"list_mu")
            temp = []
            for c in list
                mu_evol = c["list_mu"]
                indices_mu_opti = get_ind_mu_change(mu_evol)
                c_red = Dict()
                for (key,prop) in c
                    prop_red = prop[indices_mu_opti]
                    c_red[key] = prop_red
                end
                push!(temp,c_red)
            end
            push!(list_red,temp)
        else
            push!(list_red,list_evol_lists[i])
        end
        i+= 1
    end
    return list_red
end

function get_ind_mu_change(mu_evol)
    inds = []
    for i in 1:length(mu_evol)-1
        if mu_evol[i+1] < mu_evol[i]
            push!(inds,i)
        end
    end
    return inds
end

function get_dataframes_best(list_evol_lists)
    
    list_dfs = []
    for (i,list) in enumerate(list_evol_lists)
        list_cols = keys(list[1])
        @show(list_cols)
        df = DataFrame()
        for col in list_cols
            df[!,col] = Float64[]
        end

        for (c,list_c) in enumerate(list)
            if haskey(list_c,"profit_evol_val")
                idx = argmax(list_c["profit_evol_val"])
            elseif haskey(list_c, "profit_evol_val_RN")
                idx = argmax(list_c["profit_evol_val_RN"])
            else
                error("List $i does not have valid key for calculating best model for config $c")
            end

            values = []
            for col in list_cols
                push!(values,list_c[col][idx])
            end
            push!(df,values)
        end

        push!(list_dfs,df)
    end
    return list_dfs

end

function sort_df(list_dfs,type="evol")

    list_sorted = []
    list_best_configs = []
    

    for df in list_dfs
        sort_col = nothing
        if type == "evol"
            if "profit_evol_val" in names(df)
                sort_col = "profit_evol_val"
            elseif "profit_evol_val_RN" in names(df)
                sort_col = "profit_evol_val_RN"
            elseif "a_profit_val" in names(df)
                sort_col = "a_profit_val"
            elseif "a_profit_val_RN" in names(df)
                sort_col = "a_profit_val_RN"
            end
        elseif type == "outcome"
            sort_col = "a_profit_val_opt"
        end
        push!(list_best_configs,argmax(df[!,sort_col]))
        df_sorted = sort(df, order(sort_col,rev=true))
        push!(list_sorted,df_sorted)
    end



    return list_sorted,list_best_configs
end

function get_properties_best(list_df,configs,prop_str)

    keys = translate_property(prop_str)
    list_props = []

    for (i,df) in zip(configs,list_df)
        if keys[1] in names(df)
            append!(list_props,df[i,keys[1]])
        else
            append!(list_props,df[i,keys[2]])
        end
    end
    return list_props

end

function translate_property(prop)

    str_julia = nothing
    str_python = nothing

    if prop == "profit_val"
        str_julia = "profit_evol_val"
        str_python = "profit_evol_val_RN"
    elseif prop == "profit_test"
        str_julia = "profit_evol_test"
        str_python = "profit_evol_test_RN"
    elseif prop == "profit_train"
        str_julia = "profit_evol_train"
        str_python = "profit_evol_train_RN"
    elseif prop == "regret_test"
        str_julia = "regret_evol_test"
        str_python = str_julia
    elseif prop == "regret_val"
        str_julia = "regret_evol_val"
        str_python = str_julia
    end
    return [str_julia,str_python]
end

function get_outcomes(list_dirs)
    list_outcomes = []
    for dir in list_dirs
        dir,_ = locate_dir(dir)
        loc = "$(dir)/outcome.csv"
        df = CSV.File(loc) |> DataFrame

        push!(list_outcomes,df)
    end
    return list_outcomes
end

function get_property_outcome_config(list_outcomes,list_configs,prop_str)
    list_prop = []
    for (i,out) in enumerate(list_outcomes)
        prop = out[out[!,"a_config"] .== list_configs[i],prop_str][1]
        push!(list_prop,prop)
    end
    return list_prop
end

function get_property_evol(dict_evols,prop_str)
    keys_prop = translate_property(prop_str)
    prop = nothing
    if keys_prop[1] in keys(dict_evols)
        prop = dict_evols[keys_prop[1]]
    elseif keys_prop[2] in keys(dict_evols)
        prop = dict_evols[keys_prop[2]]
    else
        error("Invalid key")
    end
    return prop
end

function calc_property_outcome(list_outcomes, list_configs, prop_str, set_str, opt=false)

    list_props = []

    suffix = "" #This suffix will ensure we consider the optimized outcome instead of the best during training procedure if opt=true
    if opt
        suffix = "_opt"
    end

    for (outcome,config) in zip(list_outcomes,list_configs)

        cols = names(outcome)
        
        if prop_str == "impr_profit"
            impr = nothing
            if "a_profit_RN_$(set_str)" ∈ cols
                profit = outcome[config,"a_profit_RN_$(set_str)"]
                profit_fc = outcome[config,"a_profit_RN_$(set_str)_fc"]      
            else
                profit = outcome[config,"a_profit_$(set_str)$(suffix)"]
                profit_fc = outcome[config,"a_profit_$(set_str)_fc"]
            end
            impr = (profit-profit_fc)/profit_fc
            push!(list_props,impr)

        elseif prop_str == "regret"
            if "a_profit_RN_$(set_str)" ∈ cols
                profit = outcome[config,"a_profit_RN_$(set_str)"]
                profit_fc = outcome[config,"a_profit_RN_$(set_str)_fc"]
                profit_pf = outcome[config,"ab_profit_PF_$(set_str)"]
            
            else
                profit = outcome[config,"a_profit_$(set_str)$(suffix)"]
                profit_fc = outcome[config,"a_profit_$(set_str)_fc"]
                profit_pf = outcome[config,"ab_profit_$(set_str)_PF"]
            end
            regr = profit_pf - profit
            push!(list_props,regr)        

        elseif prop_str == "impr_regret"
            impr = nothing
            if "a_profit_RN_$(set_str)" ∈ cols
                profit = outcome[config,"a_profit_RN_$(set_str)"]
                profit_fc = outcome[config,"a_profit_RN_$(set_str)_fc"]
                profit_pf = outcome[config,"ab_profit_PF_$(set_str)"]
            
            else
                profit = outcome[config,"a_profit_$(set_str)$(suffix)"]
                profit_fc = outcome[config,"a_profit_$(set_str)_fc"]
                profit_pf = outcome[config,"ab_profit_$(set_str)_PF"]

            end
            regr = profit_pf - profit
            regr_fc = profit_pf - profit_fc
            impr = (regr_fc - regr)/regr_fc
            push!(list_props,impr)        
        end
    end

    return list_props
end

function calc_price_evolution(list_fc_lists,feat)
    price_evol = []
    act_function = nothing
    for list_evol in list_fc_lists
        price_evol_config = []
        for list in list_evol
            
            if length(list) >2
                act_function = "softplus"
                println("softplus")
            else
                act_function = "linear"
            end
            price = calc_price_from_net(list,feat,act_function)
            push!(price_evol_config,price)
        end
        push!(price_evol,price_evol_config)
    end
    return price_evol
end

function calc_subgradient(list_fc_price,act_price,OP_params_dict)
    subgrad = []
    abs_sum_subgrad = []
    opt_sched = calc_optimal_schedule(act_price,OP_params_dict,"Gurobi")

    for evol in list_fc_price
        subgrad_config = []
        abs_sum_subgrad_config = []
        for fc_price in evol
            schedule_adjusted = calc_optimal_schedule(2 .* fc_price - act_price,OP_params_dict,"Gurobi")
            sg = 2 .*(opt_sched - schedule_adjusted)
            push!(subgrad_config,sg)
            append!(abs_sum_subgrad_config,sum(abs.(sg)))
        end
        push!(subgrad,subgrad_config)
        push!(abs_sum_subgrad,abs_sum_subgrad_config)
    end
    return subgrad,abs_sum_subgrad
end

function get_profit(list_outcomes,best_configs, mode ,set_str)
    list_profit = []
    for (outcome,config) in zip(list_outcomes,best_configs)
        cols = names(outcome)
        if "a_profit_RN_$(set_str)" ∈ cols
            if mode == "PF"
                push!(list_profit,outcome[config,"ab_profit_$(mode)_$(set_str)"])
            elseif mode == "fc"
                push!(list_profit,outcome[config,"a_profit_RN_$(set_str)_$(mode)"])
            end

        else
            if mode == "PF"
                push!(list_profit,outcome[config,"ab_profit_$(set_str)_$(mode)"])
            elseif mode == "fc"
                push!(list_profit,outcome[config,"a_profit_$(set_str)_$(mode)"])
            end
        end
    end
    return list_profit
end

function get_filename_julia(config,prop,type="npz")
    #Value: choice of 'train_profit', 'val_profit', 'test_profit', 'train_profit_RA', 'val_profit_RA', 'test_profit_RA', 'mu'    

    str=nothing

    if type == "jld2"
        str = "train_evols"
    else

        if prop == "train_profit"
            str = "profitEvol_train"
        elseif prop == "val_profit"
            str = "profitEvol_val"
        elseif prop == "test_profit"
            str = "profitEvol_test"
        elseif prop == "mu"
            str = "listMu"
        end

    end


    return "config_$(config)_$(str).$(type)"
end

function get_property_value_python(config,prop,read_folder)

    properties = npzread("$(read_folder)/config_$(config)_profitEvol.npz")

    if prop == "train_profit"
        key = "array_2"
    elseif prop == "val_profit"
        key = "array_4"
    elseif prop == "test_profit"
        key = "array_6"
    elseif prop == "train_profit_RA"
        key = "array_1"
    elseif prop == "val_profit_RA"
        key = "array_3"
    elseif prop == "test_profit_RA"
        key = "array_5"
    elseif prop == "n_gradients_zero"
        key = "array_7"
    end
    return properties[key]

end

function get_prop_dict(read_codes_dict)


    prop_dict = OrderedDict()

    for key in keys(read_codes_dict)
        
        lang = read_codes_dict[key][1]
        base_folder= "./training/train_output/"
        
        read_folder = base_folder*read_codes_dict[key][2]
        config = read_codes_dict[key][3]
        prop = read_codes_dict[key][4]

        if lang == "julia"
            filename = get_filename_julia(config,prop)
            if isfile(read_folder*filename)
                property_value = npzread(read_folder*filename)
            else
                filename = get_filename_julia(config,prop,"jld2")
                property_value = JLD2.load(read_folder*filename)["train_evols"][prop]
            end

        elseif lang == "python"
            
            property_value = get_property_value_python(config,prop,read_folder)

        end


        prop_dict[key] = property_value
    end

    return prop_dict
end


