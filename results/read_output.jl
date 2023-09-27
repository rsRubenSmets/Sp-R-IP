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
    "20230920_scaled_subgradient_linear_warm",
    "20230920_scaled_subgradient_softplus_warm",
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


