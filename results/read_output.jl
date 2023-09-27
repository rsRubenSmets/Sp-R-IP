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
include("functions_support.jl")

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
    base_dir_julia = "C:/Users/u0137781/OneDrive - KU Leuven/da_scheduling/Julia scripts/trained_models/LR/LA24/SPO/"
    base_dir_python = "C:/Users/u0137781/OneDrive - KU Leuven/da_scheduling/Python scripts/trained_models/LR/LA24/torch/"

    if occursin("_ID",dir) 
        dir = base_dir_python*dir
        return dir,"h5"
    elseif occursin("_IP",dir) || occursin("_subgradient",dir)
        dir = base_dir_julia*dir
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







#### Obtaining results of Table 1  #####
list_dirs_cold = [
    "20230920_scaled_IDQ_linear_cold",
    "20230920_scaled_IDQ_softplus_cold",
    "20230920_scaled_subgradient_linear_cold",
    "20230920_scaled_subgradient_softplus_cold",
    "20230919_scaled_IP_auto_linear_cold",
    "20230920_scaled_IP_auto_softplus_cold",
    "20230919_scaled_IP_manualS_linear_cold",
    "20230919_scaled_IP_manualS_softplus_cold",
    "20230919_scaled_IP_manualD_linear_cold",
    "20230919_scaled_IP_manualD_softplus_cold",
]

list_dirs_warm = [
    "20230922_scaled_IDQ_linear_warm",
    "20230922_scaled_IDQ_softplus_warm",
    "20230919_scaled_subgradient_linear_warm",
    "20230919_scaled_subgradient_softplus_warm",
    "20230919_scaled_IP_auto_linear_warm",
    "20230919_scaled_IP_auto_softplus_warm",
    "20230919_scaled_IP_manualS_linear_warm",
    "20230919_scaled_IP_manualS_softplus_warm",
    "20230919_scaled_IP_manualD_linear_warm",
    "20230919_scaled_IP_manualD_softplus_warm",
]

list_dirs_reform = [ #Reformulation models (Sp-R) treated differently because for those, it's the validation regret for the optimized solution that counts for finding the best model
    "20230919_scaled_IP_auto_linear_cold",
    "20230920_scaled_IP_auto_softplus_cold",
    "20230919_scaled_IP_auto_linear_warm",
    "20230919_scaled_IP_auto_softplus_warm"
]



list_evols_cold,_ = read_evol_lists(list_dirs_cold,false)
list_evols_red_cold = reduce_evol_lists(list_evols_cold)
dfs_best_cold = get_dataframes_best(list_evols_red_cold)
df_best_sorted_cold,best_configs_cold = sort_df(dfs_best_cold)
test_profits_cold = get_properties_best(dfs_best_cold,best_configs_cold,"profit_test")
test_regret_cold = get_properties_best(dfs_best_cold,best_configs_cold,"regret_val")
list_outcomes_cold = get_outcomes(list_dirs_cold)
train_times_cold = get_property_outcome_config(list_outcomes_cold,best_configs_cold,"b_train_time")
test_profit_improvement_cold = calc_property_outcome(list_outcomes_cold, best_configs_cold, "impr_profit", "val")
test_regret_improvement_cold = calc_property_outcome(list_outcomes_cold, best_configs_cold, "impr_regret", "val")

list_evols_warm,_ = read_evol_lists(list_dirs_warm,false)
list_evols_red_warm = reduce_evol_lists(list_evols_warm)
dfs_best_warm = get_dataframes_best(list_evols_red_warm)
df_best_sorted_warm,best_configs_warm = sort_df(dfs_best_warm)
test_profits_warm = get_properties_best(dfs_best_warm,best_configs_warm,"profit_test")
test_regret_warm = get_properties_best(dfs_best_warm,best_configs_warm,"regret_val")
list_outcomes_warm = get_outcomes(list_dirs_warm)
train_times_warm = get_property_outcome_config(list_outcomes_warm,best_configs_warm,"b_train_time")
test_profit_improvement_warm = calc_property_outcome(list_outcomes_warm, best_configs_warm, "impr_profit", "val")
test_regret_improvement_warm = calc_property_outcome(list_outcomes_warm, best_configs_warm, "impr_regret", "val")


list_outcomes_reform = get_outcomes(list_dirs_reform)
df_best_sorted_reform, best_configs_reform = sort_df(list_outcomes_reform,"outcome")
train_times_reform = get_property_outcome_config(list_outcomes_reform,best_configs_reform,"b_train_time")
test_profits_reform = get_property_outcome_config(list_outcomes_reform,best_configs_reform,"a_profit_test_opt")
test_regret_reform = calc_property_outcome(list_outcomes_reform, best_configs_reform, "regret", "val", true)
test_profit_improvement_reform = calc_property_outcome(list_outcomes_reform, best_configs_reform, "impr_profit", "val", true)
test_regret_improvement_reform = calc_property_outcome(list_outcomes_reform, best_configs_reform, "impr_regret", "val", true)




##### Figure train and validation profit comparing 3 models #####


list_dirs = [
    "20230919_scaled_IP_manualD_linear_warm",
    "20230920_scaled_subgradient_linear_warm",
    "20230919_scaled_IDQ_linear_warmStart",
]

list_evol_lists = read_evol_lists(list_dirs)

list_evol_lists_reduced = reduce_evol_lists(list_evol_lists)

df_best = get_dataframes_best(list_evol_lists_reduced)
df_sorted_best,configs = sort_df(df_best)
 


list_outcomes = get_outcomes(list_dirs)
sorted_outcomes,_ = sort_df(list_outcomes)


configs = [[13,5,11],[24,17,30],[12,2,21]]

vis = "profit_train"

if vis == "profit_train"
    yguide = "Train profit (€)"
elseif vis == "profit_val"
    yguide = "Validation profit (€)"
end

labels = ["Sp-R-IPd","Sp-SG","ID-Q"]
colors = [:green,:blue,:orange]

plot()

for (i,list) in enumerate(list_evol_lists_reduced)

    lbl = labels[i]
    clr = colors[i]


    for (j,c) in enumerate(configs[i])
        
        dict = list[c]
        prop = get_property_evol(dict,vis)

        if j == 1
            lbl = labels[i]
        else
            lbl = ""
        end

        plot!(prop,
        label=lbl,
        color = clr,
        linewidth=5,
        xguidefontsize=40,
        yguidefontsize=40,
        xticks=:auto,xtickfontsize=30,
        yticks=:auto,ytickfontsize=30,
        xguide="Train iteration",
        yguide=yguide,
        legendfontsize=30,
        legend=:bottomright,
        size=(1500,1100),
        left_margin = 15Plots.mm,
        bottom_margin= 10Plots.mm,
        )
    end
    

end

if vis == "profit_val"
    ylims!(0.375,0.425)
elseif vis == "profit_train"
    ylims!(1.38,1.54)
end


dir = "C:/Users/u0137781/OneDrive - KU Leuven/da_scheduling/Plots/"
savefig(dir*"20230913_trainProfit_across")














##### Check out specific #####

list_dirs = [
    "20230920_scaled_subgradient_softplus_warm"
]

list_evols = read_evol_lists(list_dirs)

config = 13

mu = list_evols[1][config]["list_mu"]
train_profit = list_evols[1][config]["profit_evol_train"]
val_profit = list_evols[1][config]["profit_evol_val"]
regret_train = list_evols[1][config]["regret_evol_train"]
regret_val = list_evols[1][config]["regret_evol_val"]

plot(log10.(mu)[1:end-1],regret_val)

xflip!()

c=1
list_evols[1][c]["profit_evol_train"]
list_evols[1][c]["list_mu"]


configs = [2,7,13]
configs = [i for i in 1:30]
configs=[7,3,6,2]
configs = [27]
for i in 1:32
    if i ∉ [24,22,18,30,12,8,4,16,20,32,28,26,31,27]
        append!(configs,i)
    end
end


plot()

for c in configs
    #plot!(log10.(list_evols[1][c]["list_mu"])[1:end-1],list_evols[1][c]["profit_evol_train"],legend=:outerright,color=:black)
    plot!(list_evols[1][c]["regret_evol_train"],legend=:false,color=:black)

end

title!("Train profit")
xflip!()


3 ∉ [1,3]


bdp = "C:/Users/u0137781/OneDrive - KU Leuven/da_scheduling/Julia scripts/trained_models/LR/LA24/SPO/"
dir = bdp*"20230925_subgradient_nonlinear_test4/config_1_best_net.jld2"

check =JLD2.load(dir)["best_net"]  

OP_params_dict["activation_function"] = "softplus"

calc_profit_from_forecaster(check,OP_params_dict,features_train,labels_train[1])










##### EXAMPLE SUBGRADIENT #####


# Getting features

train_share = 1
days_train = floor(Int,1/train_share)
last_ex_test = 59 #59
repitition = 1
la=24

loc_data = "C:/Users/u0137781/OneDrive - KU Leuven/da_scheduling/Data/processed_data/SPO_DA/"

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



features_train, features_val, features_test, price_train, price_val, price_test, _ , _ = preprocess_data(data_dict)

labels_train,labels_val,labels_test = preprocess_labels(training_dict["type_train_labels"],price_train,price_val,price_test,OP_params_dict,data_dict)


list_dirs = [
    "20230924_subgradient_softplus_warm_1_59",
]


list_evols,list_fc_lists = read_evol_lists(list_dirs,true)
list_evols_red = reduce_evol_lists(list_evols)
dfs_best = get_dataframes_best(list_evols_red)
df_best_sorted,best_configs = sort_df(dfs_best)
test_profits = get_properties_best(dfs_best,best_configs,"profit_test")
test_regret = get_properties_best(dfs_best,best_configs,"regret_test")
list_outcomes = get_outcomes(list_dirs)
train_times = get_property_outcome_config(list_outcomes,best_configs,"b_train_time")
test_profit_improvement = calc_property_outcome(list_outcomes, best_configs, "impr_profit", "test")
test_regret_improvement = calc_property_outcome(list_outcomes, best_configs, "impr_regret", "test")
fc_price = calc_price_evolution(list_fc_lists,features_train)
subgradient,abs_sum_subgrad = calc_subgradient(fc_price,labels_train[1],OP_params_dict)

subgradient[16][97]
plot(abs_sum_subgrad[16])

subgradient[16][1][1:24]

plot([subgradient[16][i][1] for i in 1:200])

plot([abs_sum_subgrad[32][i] for i in 1:200])


# Generate the colormap
cmap = get_color_palette(:viridis, 200)

# Initialize the main plot
p1 = plot(legend=false)

# Plot each array
for (i, data) in enumerate(fc_price[16])
    if i > 0
        plot!(p1,transpose(data), linecolor=cmap[i], linewidth=1, label="")
    end
end

plot!(p1,transpose(labels_train[1]),linecolor=:black,linewidth = 1, label="Actual price")




# Create a dummy plot to act as a colorbar
p3 = plot([1, 1], [1, 200], linecolor=cmap, linewidth=20, legend=false, xlims=(0, 2), ylims=(0, 200), grid=false, axis=false)

# Combine the plots
p = plot(p1,p3, layout=(1, 2), size=(800, 400))

# Show the plot
display(p)





plot()

for evols in list_evols_red[1]
    plot!(evols["regret_evol_train"])
end

title!("A title")




index1 = 100
index2 = 198
config = 16

using Printf
evols = list_evols_red[1][16]

p3= plot(evols["regret_evol_train"],linewidth=3,size=(400,550),left_margin = 5Plots.mm)

# X-axis and Y-axis labels with font sizes
xaxis!("Epoch", fontsize=100, tickfont=font(16))
yaxis!("Train regret (€)", fontsize=16, tickfont=font(10))

# Custom function to format y-ticks in scientific notation
function scientific_notation(ticks)
    return [@sprintf("%.0e", tick) for tick in ticks]
end

# Generate tick positions and their labels
tick_positions = [0.0, 1e-3, 2e-3]
tick_labels = scientific_notation(tick_positions)

# Apply custom labels to y-ticks
yticks!(tick_positions, tick_labels)

# Legend font size
plot!(legend=false)
dir = "C:/Users/u0137781/OneDrive - KU Leuven/da_scheduling/Plots/"
savefig(dir*"20230925_SG_example_evol")



c_fc = -fc_price[config][index1][1:24]
c_act = -labels_train[1][1:24]
c_adj = 2 .* c_fc - c_act

w_fc = calc_optimal_schedule(-transpose(c_fc),OP_params_dict,"Gurobi")
w_act = calc_optimal_schedule(-transpose(c_act),OP_params_dict,"Gurobi")
w_adj = calc_optimal_schedule(-transpose(c_adj),OP_params_dict,"Gurobi")


p1 = plot(c_fc, label="ĉ", linewidth=3,size=(400,300))
plot!(c_act, label="c", linewidth=3)
plot!(c_adj, label="2ĉ -c", linewidth=3)

# X-axis and Y-axis labels with font sizes
xaxis!("Hour of day", fontsize=100, tickfont=font(16))
yaxis!("Opposite price (€)", fontsize=16, tickfont=font(10))

# To set xticks
plot!(xticks=(2:2:24, string.(2:2:24))) 

# Legend font size
plot!(legendfontsize=10)

dir = "C:/Users/u0137781/OneDrive - KU Leuven/da_scheduling/Plots/"
savefig(dir*"20230925_SG_example_price")



p2=plot(w_fc[49:end],label="x*(ĉ)",linewidth=3,size=(400,300))
plot!(w_act[49:end],label="x*(c)",linestyle=:dash,linewidth=3)
plot!(w_adj[49:end],label="x*(2ĉ-c)",linewidth=3)

# X-axis and Y-axis labels with font sizes
xaxis!("Hour of day", fontsize=100, tickfont=font(16))
yaxis!("State of Charge (MWh)", fontsize=16, tickfont=font(10))

# Legend font size
plot!(legendfontsize=13)

# To set xticks
plot!(xticks=(2:2:24, string.(2:2:24)))

plot!(legendfontsize=10)


dir = "C:/Users/u0137781/OneDrive - KU Leuven/da_scheduling/Plots/"
savefig(dir*"20230925_SG_example_sched")


# Custom layout
l = @layout([a b;c])
p = plot(p1, p2,p3, layout=(2, 2), size=(800, 600))

# Show the plot
display(p)


sched1 = calc_optimal_schedule(fc_price[c][index1],OP_params_dict,"Gurobi")
sched2 = calc_optimal_schedule(fc_price[c][index2],OP_params_dict, "Gurobi")

plot(fc_price[c][index1][1:24],label="price 1")
#plot!(labels_train[1][1:24]fc_price[c][index2][1:24],label="price 2")
plot!(labels_train[1][1:24],label="actual price")

plot(sched1[49:end], label="sched1")
#plot!(sched2[1:24], label="sched2")
plot!(sched_1_subgrad[49:end],label="sched opti")





plot(transpose(fc_price[1][198]),label="price fc 197")
plot(transpose(fc_price[1][198]),label="price fc 198")
plot!(transpose(labels_train[1]),label="actual price")
title!("title")


subgrad = 2 .*(calc_optimal_schedule(labels_train[1],OP_params_dict)-calc_optimal_schedule(2 .*fc_price[1][100]-labels_train[1],OP_params_dict))




index = 197
sched_1_subgrad = calc_optimal_schedule(labels_train[1],OP_params_dict,"Gurobi")
sched_2_subgrad = calc_optimal_schedule(2 .*fc_price[c][index]-labels_train[1],OP_params_dict,"Gurobi")

plot(sched_1_subgrad[1:end],label="opt sched")
plot!(sched_2_subgrad[1:end],label="adjusted sched")


plot(2 .*(sched_1_subgrad-sched_2_subgrad)[1:24])


