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
using JuMP
using Ipopt
using NPZ
using Plots
using Printf
using DataStructures
using ColorSchemes
include("../experiment/functions_support.jl")
include("functions_support_results.jl")


# function get_linestyle(lang)
#     if lang == "julia"
#         return :solid
#     elseif lang == "python"
#         return :dash
#     end
# end

# function get_color(folder)
#     if occursin("linear", folder)
#         return :blue
#     elseif occursin("softplus", folder)
#         return :green
#     elseif occursin("relu", folder)
#         return :Base.liblapack_name
#     end
# end

# function reduce_output_auto(dict_evol)
#     mu = dict_evol["mu"]

# end


##### Obtaining Figure 2 #####
#Pre-processing
list_dirs = [
    "20230919_scaled_IP_manualD_softplus_warm",
    "20230919_scaled_IP_manualS_softplus_warm",
    "20230919_scaled_IP_auto_softplus_warm",
]

list_evol_lists,_ = read_evol_lists(list_dirs,false)

list_evol_lists_reduced = reduce_evol_lists(list_evol_lists)

list_df_best_outcomes = get_dataframes_best(list_evol_lists_reduced)

_,list_best_configs = sort_df(list_df_best_outcomes)


labels = ["Sp-R-IPd","Sp-R-IPs","Sp-R-IP"]



#Figure 2a
vis = "regret_evol_train"

plot()

for (i,(config,list)) in enumerate(zip(list_best_configs,list_evol_lists_reduced))

    println(i)
    dict = list[13]
    mu = log10.(dict["list_mu"])
    prop = dict[vis]

    plot!(mu,prop,
    label=labels[i],
    marker=:circle,
    ms=10,
    linewidth=10,
    xguidefontsize=40,
    yguidefontsize=40,
    xticks=:auto,xtickfontsize=30,
    yticks=:auto,ytickfontsize=30,
    xguide="log(μ)",
    yguide="Train regret (€)",
    legendfontsize=30,
    legend=:bottomleft,
    size=(1500,1100),
    left_margin = 15Plots.mm,
    bottom_margin= 10Plots.mm,
    latex=true
    )
    

end

xflip!()


#Figure 2b
vis = "regret_evol_val"

plot()

for (i,(config,list)) in enumerate(zip(list_best_configs,list_evol_lists_reduced))

    println(i)
    dict = list[13]
    mu = log10.(dict["list_mu"])
    prop = dict[vis]

    plot!(mu,prop,
    label=labels[i],
    marker=:circle,
    ms=10,
    linewidth=10,
    xguidefontsize=40,
    yguidefontsize=40,
    xticks=:auto,xtickfontsize=30,
    yticks=:auto,ytickfontsize=30,
    xguide="log(μ)",
    yguide="Train regret (€)",
    legendfontsize=30,
    legend=:bottomleft,
    size=(1500,1100),
    left_margin = 15Plots.mm,
    bottom_margin= 10Plots.mm,
    latex=true
    )
    

end

xflip!()




##### Obtaining Figure 3 #####

#Pre-processing
dict_read_codes_3a = OrderedDict(
    #Dictionary with combination of read code, config and value to be inspected
    #Value: choice of 'train_profit', 'val_profit', 'test_profit', 'train_profit_RA', 'val_profit_RA', 'test_profit_RA', 'mu'
    "Sp-R-IP" => ["julia", "20230811_scaled_softplus_warmStart_dynamic/",11,"train_profit"],
    "γ=0.01" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",1,"train_profit"],
    "γ=0.03" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",2,"train_profit"],
    "γ=0.1" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",3,"train_profit"],
    "γ=0.3" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",4,"train_profit"],
    "γ=1" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",5,"train_profit"],
    "γ=3" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",6,"train_profit"],
    "γ=10" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",7,"train_profit"],
    "γ=30" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",8,"train_profit"],

)

dict_read_codes_3b = OrderedDict(
    #Dictionary with combination of read code, config and value to be inspected
    #Value: choice of 'train_profit', 'val_profit', 'test_profit', 'train_profit_RA', 'val_profit_RA', 'test_profit_RA', 'mu'
    
    "γ=0.01|mod" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",1,"train_profit_RA"],
    "γ=0.03|mod" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",2,"train_profit_RA"],
    "γ=0.1|mod" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",3,"train_profit_RA"],
    "γ=0.3|mod" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",4,"train_profit_RA"],
    "γ=1|mod" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",5,"train_profit_RA"],
    "γ=3|mod" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",6,"train_profit_RA"],
    "γ=10|mod" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",7,"train_profit_RA"],
    "γ=30|mod" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",8,"train_profit_RA"],

)
dict_read_codes_3c = OrderedDict(
    #Dictionary with combination of read code, config and value to be inspected
    #Value: choice of 'train_profit', 'val_profit', 'test_profit', 'train_profit_RA', 'val_profit_RA', 'test_profit_RA', 'mu'
    "Sp-R-IP" => ["julia", "20230811_scaled_softplus_warmStart_dynamic/",11,"val_profit"],
    "γ=0.01" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",1,"val_profit"],
    "γ=0.03" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",2,"val_profit"],
    "γ=0.1" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",3,"val_profit"],
    "γ=0.3" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",4,"val_profit"],
    "γ=1" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",5,"val_profit"],
    "γ=3" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",6,"val_profit"],
    "γ=10" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",7,"val_profit"],
    "γ=30" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",8,"val_profit"],

)

dict_read_codes_3d = OrderedDict(
    #Dictionary with combination of read code, config and value to be inspected
    #Value: choice of 'train_profit', 'val_profit', 'test_profit', 'train_profit_RA', 'val_profit_RA', 'test_profit_RA', 'mu'
    
    "γ=0.01|mod" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",1,"val_profit_RA"],
    "γ=0.03|mod" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",2,"val_profit_RA"],
    "γ=0.1|mod" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",3,"val_profit_RA"],
    "γ=0.3|mod" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",4,"val_profit_RA"],
    "γ=1|mod" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",5,"val_profit_RA"],
    "γ=3|mod" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",6,"val_profit_RA"],
    "γ=10|mod" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",7,"val_profit_RA"],
    "γ=30|mod" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",8,"val_profit_RA"],

)


colors = palette([:blue,:yellow],length(dict_read_codes_orig))

profit_dict_3a = get_prop_dict(dict_read_codes_3a)
profit_dict_3b = get_prop_dict(dict_read_codes_3b)
profit_dict_3c = get_prop_dict(dict_read_codes_3c)
profit_dict_3d = get_prop_dict(dict_read_codes_3d)

profit_dict_mod = get_prop_dict(dict_read_codes_mod)

profit_val_pf = 0.489199348469553
profit_test_pf = 0.515363768
profit_train_pf = 1.754977235




#Figure 3a
plot()

for (i,lbl) in enumerate(keys(profit_dict_3a))

    linestyle=:solid
    color=nothing
    lw=nothing
    if lbl == "Sp-R-IP"
        color = "black"
        lw=2.0
    else
        color = colors[i]
        lw=1.5
    end
    plot!([profit_train_pf-profit_dict_3a[lbl][i] for i in 1:length(profit_dict_3a[lbl])],label=lbl,linestyle=linestyle,color=color,linewidth=lw)

end

yaxis!("Regret (€)")
xaxis!("Training iteration")


#Figure 3b
plot()

for (i,lbl) in enumerate(keys(profit_dict_3b))

    linestyle=:solid
    color=nothing
    lw=nothing
    if lbl == "Sp-R-IP"
        color = "black"
        lw=2.0
    else
        color = colors[i]
        lw=1.5
    end
    plot!([profit_train_pf-profit_dict_3b[lbl][i] for i in 1:length(profit_dict_3b[lbl])],label=lbl,linestyle=linestyle,color=color,linewidth=lw)

end

yaxis!("Regret (€)")
xaxis!("Training iteration")


#Figure 3c
plot()

for (i,lbl) in enumerate(keys(profit_dict_3c))

    linestyle=:solid
    color=nothing
    lw=nothing
    if lbl == "Sp-R-IP"
        color = "black"
        lw=2.0
    else
        color = colors[i]
        lw=1.5
    end
    plot!([profit_val_pf-profit_dict_3c[lbl][i] for i in 1:length(profit_dict_3c[lbl])],label=lbl,linestyle=linestyle,color=color,linewidth=lw)

end

yaxis!("Regret (€)")
xaxis!("Training iteration")


#Figure 3d
plot()

for (i,lbl) in enumerate(keys(profit_dict_3d))

    linestyle=:solid
    color=nothing
    lw=nothing
    if lbl == "Sp-R-IP"
        color = "black"
        lw=2.0
    else
        color = colors[i]
        lw=1.5
    end
    plot!([profit_val_pf-profit_dict_3d[lbl][i] for i in 1:length(profit_dict_3d[lbl])],label=lbl,linestyle=linestyle,color=color,linewidth=lw)

end

yaxis!("Regret (€)")
xaxis!("Training iteration")





##### Obtaining Figure 3 #####

# Getting features

train_share = 1
days_train = floor(Int,1/train_share)
last_ex_test = 59 #59
repitition = 1
la=24
factor_size_ESS = 1

loc_data = "./data/processed_data/SPO_DA/"

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





features_train, features_val, features_test, price_train, price_val, price_test, _ , _ = preprocess_data(data_dict)

labels_train,labels_val,labels_test = preprocess_labels(training_dict["type_train_labels"],price_train,price_val,price_test,OP_params_dict,data_dict)


list_dirs = [
    "20230924_subgradient_softplus_warm_1_59",
]


list_evols,list_fc_lists = read_evol_lists(list_dirs,true)
list_evols_red = reduce_evol_lists(list_evols)
# dfs_best = get_dataframes_best(list_evols_red)
# df_best_sorted,best_configs = sort_df(dfs_best)
# test_profits = get_properties_best(dfs_best,best_configs,"profit_test")
# test_regret = get_properties_best(dfs_best,best_configs,"regret_test")
# list_outcomes = get_outcomes(list_dirs)
# train_times = get_property_outcome_config(list_outcomes,best_configs,"b_train_time")
# test_profit_improvement = calc_property_outcome(list_outcomes, best_configs, "impr_profit", "test")
# test_regret_improvement = calc_property_outcome(list_outcomes, best_configs, "impr_regret", "test")
fc_price = calc_price_evolution(list_fc_lists,features_train)
# subgradient,abs_sum_subgrad = calc_subgradient(fc_price,labels_train[1],OP_params_dict)

index1 = 100
index2 = 198
config = 16

evols = list_evols_red[1][16]


c_fc = -fc_price[config][index1][1:24]
c_act = -labels_train[1][1:24]
c_adj = 2 .* c_fc - c_act

w_fc = calc_optimal_schedule(-transpose(c_fc),OP_params_dict,"Gurobi")
w_act = calc_optimal_schedule(-transpose(c_act),OP_params_dict,"Gurobi")
w_adj = calc_optimal_schedule(-transpose(c_adj),OP_params_dict,"Gurobi")



#Figure 4a
p1 = plot(c_fc, label="ĉ", linewidth=3,size=(400,300))
plot!(c_act, label="c", linewidth=3)
plot!(c_adj, label="2ĉ -c", linewidth=3)
xaxis!("Hour of day", fontsize=100, tickfont=font(16))
yaxis!("Opposite price (€)", fontsize=16, tickfont=font(10))
plot!(xticks=(2:2:24, string.(2:2:24))) 

plot!(legendfontsize=10)



#Figure 4b
p2=plot(w_fc[49:end],label="x*(ĉ)",linewidth=3,size=(400,300))

plot!(w_act[49:end],label="x*(c)",linestyle=:dash,linewidth=3)
plot!(w_adj[49:end],label="x*(2ĉ-c)",linewidth=3)
xaxis!("Hour of day", fontsize=100, tickfont=font(16))
yaxis!("State of Charge (MWh)", fontsize=16, tickfont=font(10))
plot!(legendfontsize=13)
plot!(xticks=(2:2:24, string.(2:2:24)))

plot!(legendfontsize=10)


#Figure 4c
p3= plot(evols["regret_evol_train"],linewidth=3,size=(400,550),left_margin = 5Plots.mm)

xaxis!("Epoch", fontsize=100, tickfont=font(16))
yaxis!("Train regret (€)", fontsize=16, tickfont=font(10))

function scientific_notation(ticks)
    return [@sprintf("%.0e", tick) for tick in ticks]
end

tick_positions = [0.0, 1e-3, 2e-3]
tick_labels = scientific_notation(tick_positions)
yticks!(tick_positions, tick_labels)

plot!(legend=false)







































dict_read_codes = OrderedDict(
    #Dictionary with combination of read code, config and value to be inspected
    #Value: choice of 'train_profit', 'val_profit', 'test_profit', 'train_profit_RA', 'val_profit_RA', 'test_profit_RA', 'mu'
    "SPO|linear" => ["julia", "20230816_scaled_linear_warmStart_mu/",5,"train_profit"],
    "SPO|softplus" => ["julia", "20230816_scaled_linear_warmStart_mu/",5,"train_profit"],
    "ID|linear" => ["python","20230816_scaled_linear_warmStart_mu",1,"train_profit"],
    "ID|softplus" => ["python","20230816_scaled_linear_warmStart_mu",1,"train_profit"],
)

profit_dict = get_prop_dict(dict_read_codes)

plot()
for (i,lbl) in enumerate(keys(profit_dict))
    linestyle = get_linestyle(dict_read_codes[lbl][1])
    #color = get_color(dict_read_codes[lbl][2])
    plot!(profit_dict[lbl],label=lbl,linestyle=linestyle)
end

title!("Test profit over training iterations")
yaxis!("Profit (€)")
xaxis!("Training iteration")


dict_read_codes = OrderedDict(
    #Dictionary with combination of read code, config and value to be inspected
    #Value: choice of 'train_profit', 'val_profit', 'test_profit', 'train_profit_RA', 'val_profit_RA', 'test_profit_RA', 'mu'
    "SPO|linear" => ["julia", "20230816_scaled_linear_warmStart_mu/",5,"train_profit"],
    "SPO|softplus" => ["julia", "20230816_scaled_linear_warmStart_mu/",5,"train_profit"],
    "ID|linear" => ["python","20230816_scaled_linear_warmStart_mu",1,"train_profit"],
    "ID|softplus" => ["python","20230816_scaled_linear_warmStart_mu",1,"train_profit"],
)

dict_read_codes_WS = OrderedDict(
    #Dictionary with combination of read code, config and value to be inspected
    #Value: choice of 'train_profit', 'val_profit', 'test_profit', 'train_profit_RA', 'val_profit_RA', 'test_profit_RA', 'mu'
    "SPO|linear" => ["julia", "20230811_scaled_linear_warmStart_dynamic/",19,"train_profit"],
    "SPO|softplus" => ["julia", "20230811_scaled_softplus_warmStart_dynamic/",19,"train_profit"],
    "ID|linear" => ["python","20230815_scaled_linear_warmStart_gamma_long_patience50_2/",32,"train_profit"],
    "ID|softplus" => ["python","20230815_scaled_softplus_warmStart_gamma_long/",47,"train_profit"],
)

dict_read_codes_noWS = OrderedDict(
    #Dictionary with combination of read code, config and value to be inspected
    #Value: choice of 'train_profit', 'val_profit', 'test_profit', 'train_profit_RA', 'val_profit_RA', 'test_profit_RA', 'mu'
    "SPO|linear" => ["julia", "20230811_scaled_linear_noWarmStart_dynamic/",13,"val_profit"],
    "SPO|softplus" => ["julia", "20230811_scaled_softplus_noWarmStart_dynamic/",13,"val_profit"],
    #"ID|linear" => ["python","20230815_scaled_linear_noWarmStart/",5,"val_profit"],
    #"ID|softplus" => ["python","20230815_scaled_softplus_noWarmStart/",7,"val_profit"],
)




profit_dict_WS = get_prop_dict(dict_read_codes_WS)
profit_dict_noWS = get_prop_dict(dict_read_codes_noWS)


plot()
for (i,lbl) in enumerate(keys(profit_dict_WS))
    linestyle = get_linestyle(dict_read_codes_WS[lbl][1])
    color = get_color(dict_read_codes_WS[lbl][2])
    plot!(profit_dict_WS[lbl],label=lbl,linestyle=linestyle,color=color)

end

yaxis!("Profit (€)")
xaxis!("Training iteration")




dict_read_codes_test = OrderedDict(
    #Dictionary with combination of read code, config and value to be inspected
    #Value: choice of 'train_profit', 'val_profit', 'test_profit', 'train_profit_RA', 'val_profit_RA', 'test_profit_RA', 'mu'
    "cyclic" => ["python", "20230823_scaled_softplus_largeESS_cyclicBC/",1,"train_profit"],
    "no_cyclicBC" => ["python", "20230823_scaled_softplus_largeESS_cyclicBC/",1,"train_profit"]
)

profit_dict_test = get_prop_dict(dict_read_codes_test)

dict_read_codes_gamma1 = OrderedDict()
dict_read_codes_gamma100 =OrderedDict()

n_configs = 8

for i in 1:n_configs
    #dict_read_codes_gamma1["gamma1_$i"] = ["python", "20230823_scaled_softplus_largeESS_noCyclicBC_gamma1/",i,"train_profit"]
    dict_read_codes_gamma100["gamma100_$i"] = ["python", "20230823_scaled_softplus_largeESS_noCyclicBC/",i,"train_profit"]
end

profit_dict_gamma1 = get_prop_dict(dict_read_codes_gamma1)
profit_dict_gamma100 = get_prop_dict(dict_read_codes_gamma100)



plot()

# for (i,lbl) in enumerate(keys(profit_dict_gamma1))
#     #linestyle = get_linestyle(dict_read_codes[lbl][1])
#     linestyle=:solid
#     plot!(profit_dict_gamma1[lbl],label=lbl,linestyle=linestyle,color="red")

# end

for (i,lbl) in enumerate(keys(profit_dict_gamma100))
    #linestyle = get_linestyle(dict_read_codes[lbl][1])
    linestyle=:solid
    plot!(profit_dict_gamma100[lbl],label=lbl,linestyle=linestyle,color="green")

end

title!("Train profit over training iterations")
yaxis!("Profit (€)")
xaxis!("Training iteration")







regs = [0,0.001,0.1,10]

dict_read_codes =OrderedDict()

dict_read_codes_NWS = OrderedDict()

for i in 1:4
    config = 3+(i-1)*8
    println(config)
    println("reg:$(regs[i])")
    dict_read_codes["reg:$(regs[i])"] = ["julia", "20230811_scaled_softplus_warmStart_dynamic/",config,"train_profit"]
    dict_read_codes_NWS["reg:$(regs[i])"] = ["julia", "20230811_scaled_linear_noWarmStart_dynamic/",config,"train_profit"]
end


profit_dict = get_prop_dict(dict_read_codes)
profit_dict_NWS = get_prop_dict(dict_read_codes_NWS)



colors=[:red,:orange,:yellow,:green]
plot()

for (i,lbl) in enumerate(keys(profit_dict))
    #linestyle = get_linestyle(dict_read_codes[lbl][1])
    linestyle=:solid
    plot!(profit_dict[lbl],label=lbl,linestyle=linestyle,color=colors[i])

end




for (i,lbl) in enumerate(keys(profit_dict_NWS))
    #linestyle = get_linestyle(dict_read_codes[lbl][1])
    color = colors[i]
    #plot!(profit_dict[lbl],label=lbl,linestyle=linestyle,color=color)
    linestyle=:dash
    plot!(profit_dict_NWS[lbl],label=lbl,linestyle=linestyle,color=colors[i])

end

title!("Train profit over training iterations")
yaxis!("Profit (€)")
xaxis!("Training iteration")






gamma  = [0.01,0.1,1,10]

dict_read_codes =OrderedDict()

for i in 1:4
    config = 1+(i-1)*2
    dict_read_codes["gamma:$(gamma[i])"] = ["python", "20230815_scaled_softplus_noWarmStart/",config,"val_profit"]
end

profit_dict = get_prop_dict(dict_read_codes)

colors=[:red,:orange,:yellow,:green]
plot()

for (i,lbl) in enumerate(keys(profit_dict))
    #linestyle = get_linestyle(dict_read_codes[lbl][1])
    linestyle=:solid
    plot!(profit_dict[lbl],label=lbl,linestyle=linestyle,color=colors[i])

end
title!("Train profit over training iterations")
yaxis!("Profit (€)")
xaxis!("Training iteration")









dict_read_codes_orig = OrderedDict(
    #Dictionary with combination of read code, config and value to be inspected
    #Value: choice of 'train_profit', 'val_profit', 'test_profit', 'train_profit_RA', 'val_profit_RA', 'test_profit_RA', 'mu'
    "Sp-R-IP" => ["julia", "20230811_scaled_softplus_warmStart_dynamic/",11,"val_profit"],
    "γ=0.01" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",1,"val_profit"],
    "γ=0.03" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",2,"val_profit"],
    "γ=0.1" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",3,"val_profit"],
    "γ=0.3" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",4,"val_profit"],
    "γ=1" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",5,"val_profit"],
    "γ=3" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",6,"val_profit"],
    "γ=10" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",7,"val_profit"],
    "γ=30" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",8,"val_profit"],

)

dict_read_codes_mod = OrderedDict(
    #Dictionary with combination of read code, config and value to be inspected
    #Value: choice of 'train_profit', 'val_profit', 'test_profit', 'train_profit_RA', 'val_profit_RA', 'test_profit_RA', 'mu'
    
    "γ=0.01|mod" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",1,"train_profit_RA"],
    "γ=0.03|mod" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",2,"train_profit_RA"],
    "γ=0.1|mod" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",3,"train_profit_RA"],
    "γ=0.3|mod" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",4,"train_profit_RA"],
    "γ=1|mod" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",5,"train_profit_RA"],
    "γ=3|mod" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",6,"train_profit_RA"],
    "γ=10|mod" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",7,"train_profit_RA"],
    "γ=30|mod" => ["python", "20230815_scaled_softplus_warmStart_gamma_long_patience100",8,"train_profit_RA"],

)


using ColorSchemes
colors = get_color_palette(:viridis, 2)
colors = palette([:blue,:yellow],length(dict_read_codes_orig))
#colors = get(ColorSchemes.rainbow, LinRange(0,1,9))


profit_dict_orig = get_prop_dict(dict_read_codes_orig)
profit_dict_mod = get_prop_dict(dict_read_codes_mod)

profit_val_pf = 0.489199348469553
profit_test_pf = 0.515363768
profit_train_pf = 1.754977235



#colors=[:orange,:red,:blue]
plot()

for (i,lbl) in enumerate(keys(profit_dict_orig))

    linestyle=:solid
    color=nothing
    lw=nothing
    if lbl == "Sp-R-IP"
        @show(lbl)
        color = "black"
        lw=2.0
    else
        color = colors[i]
        lw=1.5
    end
    plot!([profit_val_pf-profit_dict_orig[lbl][i] for i in 1:length(profit_dict_orig[lbl])],label=lbl,linestyle=linestyle,color=color,linewidth=lw)

end

yaxis!("Regret (€)")
xaxis!("Training iteration")


dir = "C:/Users/u0137781/OneDrive - KU Leuven/da_scheduling/Plots/"
savefig(dir*"20230925_ID_gammaComp_val_mod")



plot()
for (i,lbl) in enumerate(keys(profit_dict_mod))
    #linestyle = get_linestyle(dict_read_codes[lbl][1])
    #color = get_color(dict_read_codes[lbl][2])
    #plot!(profit_dict[lbl],label=lbl,linestyle=linestyle,color=color)
    linestyle=:dash
    plot!(profit_dict_mod[lbl],label=lbl,linestyle=linestyle,color=colors[i])

end

title!("Validation profit over training iterations")
yaxis!("Profit (€)")
xaxis!("Training iteration")







##### Reduction of dimensions auto & get dataframe best results per config #####

function read_evol_lists(list_dirs)
    base_dir = "C:/Users/u0137781/OneDrive - KU Leuven/da_scheduling/Julia scripts/trained_models/LR/LA24/SPO/"
    list_evol_lists = []
    for dir in list_dirs
        dir = base_dir*dir
        list_evols = []
        c = 1
        loc = "$(dir)/config_$(c)_train_evols.jld2"
        println(loc)
        while isfile(loc)
            push!(list_evols,JLD2.load(loc))
            c+=1
            loc = "$(dir)/config_$(c)_train_evols.jld2"
        end
        push!(list_evol_lists,list_evols)
    end
    return list_evol_lists
end

function reduce_evol_lists(list_evol_lists)
    i=0
    for list in list_evol_lists
        i+= 1
        println(i)
        for c in list
            mu_evol = c["train_evols"]["list_mu"]
            indices_mu_opti = get_ind_mu_change(mu_evol)
            c["train_evols_red"] = Dict()
            for (key,prop) in c["train_evols"]
                prop_red = prop[indices_mu_opti]
                c["train_evols_red"][key] = prop_red
            end
        end
    end
    return list_evol_lists
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

function get_dataframes_best(list_evol_lists,type="full")
    
    str_evol = nothing
    if type == "full"
        str_evol = "train_evols"
    elseif type == "red"
        str_evol = "train_evols_red"
    end
    println(str_evol)
    list_dfs = []
    for (i,list) in enumerate(list_evol_lists)
        println(i)
        list_cols = keys(list[1][str_evol])
        @show(list_cols)
        df = DataFrame()
        for col in list_cols
            df[!,col] = Float64[]
        end

        for (c,list_c) in enumerate(list)
            idx = argmax(list_c[str_evol]["profit_evol_val"])
            values = []
            for col in list_cols
                push!(values,list_c[str_evol][col][idx])
            end
            push!(df,values)
        end

        push!(list_dfs,df)
    end
    return list_dfs

end

function sort_df(list_dfs,sort_col = "profit_evol_val")

    list_sorted = []
    list_best_configs = []

    for df in list_dfs
        push!(list_best_configs,argmax(df[!,sort_col]))
        df_sorted = sort(df, order(sort_col,rev=true))
        push!(list_sorted,df_sorted)
    end

    return list_sorted,list_best_configs
end


list_dirs = [
    "20230901_scaled_subgradient_linear_warmStart",
    "20230901_scaled_subgradient_linear_noWarmStart",
    "20230901_scaled_subgradient_softplus_warmStart",
    "20230901_scaled_subgradient_softplus_noWarmStart",
    "20230913_scaled_auto_linear_warmStart",
    "20230913_scaled_auto_linear_noWarmStart",
    "20230913_scaled_auto_softplus_warmStart",
    #"20230913_scaled_auto_softplus_noWarmStart",
    "20230913_scaled_manualS_linear_warmStart",
    "20230913_scaled_manualS_linear_noWarmStart",
    "20230913_scaled_manualS_softplus_warmStart",
    #"20230913_scaled_manualS_softplus_noWarmStart",
    "20230913_scaled_manualD_linear_warmStart",
    "20230913_scaled_manualD_linear_noWarmStart",
    "20230913_scaled_manualD_softplus_warmStart",
    "20230913_scaled_manualD_softplus_noWarmStart",
]




loc = "C:/Users/u0137781/OneDrive - KU Leuven/da_scheduling/Python scripts/trained_models/LR/LA24/torch/20230813_profitEvol_softplus/"

properties = npzread(loc*"config_1_profitEvol.npz")

properties["array_2"]

loc = "C:/Users/u0137781/OneDrive - KU Leuven/da_scheduling/Julia scripts/trained_models/LR/LA24/SPO/20230912_auto_test/"

properties = JLD2.load(loc*"config_1_train_evols.jld2")

x = properties["train_evols"]


list_dirs = [
    "20230913_scaled_subgradient_linear_warmStart",
    "20230913_scaled_subgradient_linear_noWarmStart",
    "20230913_scaled_subgradient_softplus_warmStart",
    "20230913_scaled_subgradient_softplus_noWarmStart",
    #"20230913_scaled_manualS_linear_warmStart",
    #"20230913_scaled_manualS_linear_noWarmStart",
    #"20230913_scaled_manualS_softplus_warmStart",
    #"20230913_scaled_manualS_softplus_noWarmStart",
    "20230913_scaled_manualD_linear_warmStart",
    "20230913_scaled_manualD_linear_noWarmStart",
    "20230913_scaled_manualD_softplus_warmStart",
    "20230913_scaled_manualD_softplus_noWarmStart",

]

list_evol_lists = read_evol_lists(list_dirs)

list_df_best_outcomes = get_dataframes_best(list_evol_lists,"full")

list_df_best_outcomes_sorted,list_best_configs = sort_df(list_df_best_outcomes)

dict_test_outcome = OrderedDict()

for (i,dir) in enumerate(list_dirs)
    println(dir)
    dict_test_outcome[dir] = list_df_best_outcomes_sorted[i][1,"profit_evol_val"]
end

dict_test_outcome

#Relative profit
profit_initial = 0.4205
dict_test_oucome_rel = OrderedDict()
for key in keys(dict_test_outcome)
    dict_test_oucome_rel[key] = round((dict_test_outcome[key]/profit_initial-1)*100,digits=1)
end

dict_test_oucome_rel



##### Plot profit vs mu #####

list_dirs = [
    # "2023091_scaled_IP_manualD_softplus_warmStart",
    # "20230913_scaled_IP_manualS_softplus_warmStart",
    # "20230913_scaled_IP_auto_softplus_warmStart",
    "20230919_scaled_IP_manualD_softplus_warm",
    "20230919_scaled_IP_manualS_softplus_warm",
    "20230919_scaled_IP_auto_softplus_warm",
]

list_evol_lists,_ = read_evol_lists(list_dirs)

list_evol_lists_reduced = reduce_evol_lists(list_evol_lists)

list_df_best_outcomes = get_dataframes_best(list_evol_lists_reduced)

_,list_best_configs = sort_df(list_df_best_outcomes)

vis = "regret_evol_train"

labels = ["Sp-R-IPd","Sp-R-IPs","Sp-R-IP"]

plot()

for (i,(config,list)) in enumerate(zip(list_best_configs,list_evol_lists_reduced))

    println(i)
    dict = list[13]
    mu = log10.(dict["list_mu"])
    prop = dict[vis]

    plot!(mu,prop,
    label=labels[i],
    marker=:circle,
    ms=10,
    linewidth=10,
    xguidefontsize=40,
    yguidefontsize=40,
    xticks=:auto,xtickfontsize=30,
    yticks=:auto,ytickfontsize=30,
    xguide="log(μ)",
    yguide="Train regret (€)",
    legendfontsize=30,
    legend=:bottomleft,
    size=(1500,1100),
    left_margin = 15Plots.mm,
    bottom_margin= 10Plots.mm,
    latex=true
    )
    

end


xflip!()


dir = "C:/Users/u0137781/OneDrive - KU Leuven/da_scheduling/Plots/"
savefig(dir*"20230925_trainRegret_vs_mu")






##### OVERALL RESULTS ##### 



dict_read_codes_cold = OrderedDict(
    #Dictionary with combination of read code, config and value to be inspected
    #Value: choice of 'train_profit', 'val_profit', 'test_profit', 'train_profit_RA', 'val_profit_RA', 'test_profit_RA', 'mu'
    "Sp-SG|linear" => ["julia", "20230901_scaled_subgradient_linear_noWarmStart/",13,"profit_evol_train"],
    "Sp-SG|softplus" => ["julia", "20230901_scaled_subgradient_softplus_noWarmStart/",25,"profit_evol_train"],
    "Sp-IPd|linear" => ["julia", "20230912_scaled_manualD_linear_noWarmStart/",19,"profit_evol_train"],
    "Sp-IPd|softplus" => ["julia", "20230912_scaled_manualD_softplus_noWarmStart/",19,"profit_evol_train"],
)

dict_read_codes_warm = OrderedDict(
    #Dictionary with combination of read code, config and value to be inspected
    #Value: choice of 'train_profit', 'val_profit', 'test_profit', 'train_profit_RA', 'val_profit_RA', 'test_profit_RA', 'mu'
    "Sp-SG|linear" => ["julia", "20230901_scaled_subgradient_linear_warmStart/",10,"profit_evol_train"],
    "Sp-SG|softplus" => ["julia", "20230901_scaled_subgradient_softplus_warmStart/",32,"profit_evol_train"],
    "Sp-IPd|linear" => ["julia", "20230912_scaled_manualD_linear_warmStart/",3,"profit_evol_train"],
    "Sp-IPd|softplus" => ["julia", "20230912_scaled_manualD_softplus_warmStart/",5,"profit_evol_train"],
)



profit_dict_warm = get_prop_dict(dict_read_codes_warm)
profit_dict_cold = get_prop_dict(dict_read_codes_cold)


plot()

for (i,lbl) in enumerate(keys(profit_dict_warm))
    if startswith(lbl,"Sp-SG")
        println("Success")
        linestyle = :dash
    elseif startswith(lbl,"Sp-IPd")
        linestyle = :solid
    end

    if endswith(lbl,"linear")
        color = :blue
    elseif endswith(lbl,"softplus")
        color = :orange
    end

    plot!(profit_dict_warm[lbl],label=lbl,linestyle=linestyle,color=color)

end

yaxis!("Profit (€)")
xaxis!("Training iteration")


dir = "C:/Users/u0137781/OneDrive - KU Leuven/da_scheduling/Plots/"
#savefig(dir*"20230913_train_warm")





for (i,lbl) in enumerate(keys(profit_dict_NWS))
    #linestyle = get_linestyle(dict_read_codes[lbl][1])
    color = colors[i]
    #plot!(profit_dict[lbl],label=lbl,linestyle=linestyle,color=color)
    linestyle=:dash
    plot!(profit_dict_NWS[lbl],label=lbl,linestyle=linestyle,color=colors[i])

end






###### FETCH TRAIN TIMES ######

# Julia


function read_outcome(list_dirs)
    base_dir = "C:/Users/u0137781/OneDrive - KU Leuven/da_scheduling/Julia scripts/trained_models/LR/LA24/SPO/"
    list_outcomes = []
    for dir in list_dirs
        dir = base_dir*dir
        loc = "$(dir)/outcome.csv"
        df = CSV.File(loc) |> DataFrame
        push!(list_outcomes,df)
    end
    return list_outcomes
end


list_outcome = read_outcome(list_dirs)

prop = "b_train_time"
sort_col = "a_profit_val"

sorted_dicts, best_configs = sort_df(list_outcome,sort_col)

dict_test_outcome = OrderedDict()

for (i,dir) in enumerate(list_dirs)
    println(dir)
    dict_test_outcome[dir] = sorted_dicts[i][1,"b_train_time"]
end

dict_test_outcome

