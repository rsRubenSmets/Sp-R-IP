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




##### Obtaining Figure 2 #####
#Pre-processing
list_dirs = [
    "Sp_R_IPd_softplus_warm",
    "Sp_R_IPs_softplus_warm",
    "Sp_R_IP_softplus_warm",
]

list_evol_lists,_ = read_evol_lists(list_dirs,false)

list_evol_lists_reduced = reduce_evol_lists(list_evol_lists)

list_df_best_outcomes = get_dataframes_best(list_evol_lists_reduced)

_,list_best_configs = sort_df(list_df_best_outcomes)


labels = ["Sp-R-IPd","Sp-R-IPs","Sp-R-IP"]



#Figure 2a
vis = "regret_evol_train"
c = 13

plot()

for (i,(config,list)) in enumerate(zip(list_best_configs,list_evol_lists_reduced))

    println(i)
    dict = list[c]
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
    dict = list[c]
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
    yguide="Validation regret (€)",
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
    "Sp-R-IP" => ["julia", "Sp_R_IPd_softplus_warm_fig3/",11,"train_profit"],
    "γ=0.01" => ["python", "IDQ_softplus_warm_fig3",1,"train_profit"],
    "γ=0.03" => ["python", "IDQ_softplus_warm_fig3",2,"train_profit"],
    "γ=0.1" => ["python", "IDQ_softplus_warm_fig3",3,"train_profit"],
    "γ=0.3" => ["python", "IDQ_softplus_warm_fig3",4,"train_profit"],
    "γ=1" => ["python", "IDQ_softplus_warm_fig3",5,"train_profit"],
    "γ=3" => ["python", "IDQ_softplus_warm_fig3",6,"train_profit"],
    "γ=10" => ["python", "IDQ_softplus_warm_fig3",7,"train_profit"],
    "γ=30" => ["python", "IDQ_softplus_warm_fig3",8,"train_profit"],

)

dict_read_codes_3b = OrderedDict(
    #Dictionary with combination of read code, config and value to be inspected
    #Value: choice of 'train_profit', 'val_profit', 'test_profit', 'train_profit_RA', 'val_profit_RA', 'test_profit_RA', 'mu'
    
    "γ=0.01|mod" => ["python", "IDQ_softplus_warm_fig3",1,"train_profit_RA"],
    "γ=0.03|mod" => ["python", "IDQ_softplus_warm_fig3",2,"train_profit_RA"],
    "γ=0.1|mod" => ["python", "IDQ_softplus_warm_fig3",3,"train_profit_RA"],
    "γ=0.3|mod" => ["python", "IDQ_softplus_warm_fig3",4,"train_profit_RA"],
    "γ=1|mod" => ["python", "IDQ_softplus_warm_fig3",5,"train_profit_RA"],
    "γ=3|mod" => ["python", "IDQ_softplus_warm_fig3",6,"train_profit_RA"],
    "γ=10|mod" => ["python", "IDQ_softplus_warm_fig3",7,"train_profit_RA"],
    "γ=30|mod" => ["python", "IDQ_softplus_warm_fig3",8,"train_profit_RA"],

)
dict_read_codes_3c = OrderedDict(
    #Dictionary with combination of read code, config and value to be inspected
    #Value: choice of 'train_profit', 'val_profit', 'test_profit', 'train_profit_RA', 'val_profit_RA', 'test_profit_RA', 'mu'
    "Sp-R-IP" => ["julia", "Sp_R_IPd_softplus_warm_fig3/",11,"val_profit"],
    "γ=0.01" => ["python", "IDQ_softplus_warm_fig3",1,"val_profit"],
    "γ=0.03" => ["python", "IDQ_softplus_warm_fig3",2,"val_profit"],
    "γ=0.1" => ["python", "IDQ_softplus_warm_fig3",3,"val_profit"],
    "γ=0.3" => ["python", "IDQ_softplus_warm_fig3",4,"val_profit"],
    "γ=1" => ["python", "IDQ_softplus_warm_fig3",5,"val_profit"],
    "γ=3" => ["python", "IDQ_softplus_warm_fig3",6,"val_profit"],
    "γ=10" => ["python", "IDQ_softplus_warm_fig3",7,"val_profit"],
    "γ=30" => ["python", "IDQ_softplus_warm_fig3",8,"val_profit"],

)

dict_read_codes_3d = OrderedDict(
    #Dictionary with combination of read code, config and value to be inspected
    #Value: choice of 'train_profit', 'val_profit', 'test_profit', 'train_profit_RA', 'val_profit_RA', 'test_profit_RA', 'mu'
    
    "γ=0.01|mod" => ["python", "IDQ_softplus_warm_fig3",1,"val_profit_RA"],
    "γ=0.03|mod" => ["python", "IDQ_softplus_warm_fig3",2,"val_profit_RA"],
    "γ=0.1|mod" => ["python", "IDQ_softplus_warm_fig3",3,"val_profit_RA"],
    "γ=0.3|mod" => ["python", "IDQ_softplus_warm_fig3",4,"val_profit_RA"],
    "γ=1|mod" => ["python", "IDQ_softplus_warm_fig3",5,"val_profit_RA"],
    "γ=3|mod" => ["python", "IDQ_softplus_warm_fig3",6,"val_profit_RA"],
    "γ=10|mod" => ["python", "IDQ_softplus_warm_fig3",7,"val_profit_RA"],
    "γ=30|mod" => ["python", "IDQ_softplus_warm_fig3",8,"val_profit_RA"],

)


colors = palette([:blue,:yellow],length(dict_read_codes_3a))

profit_dict_3a = get_prop_dict(dict_read_codes_3a)
profit_dict_3b = get_prop_dict(dict_read_codes_3b)
profit_dict_3c = get_prop_dict(dict_read_codes_3c)
profit_dict_3d = get_prop_dict(dict_read_codes_3d)

profit_val_pf = 0.489199348469553
profit_test_pf = 0.515363768
profit_train_pf = 1.754977235



llww=5
#Figure 3a
plot(
xguidefontsize=40,
yguidefontsize=40,
xticks=:auto,xtickfontsize=30,
yticks=:auto,ytickfontsize=30,
legendfontsize=25,
size=(1500,1100),
left_margin = 15Plots.mm,
bottom_margin= 10Plots.mm,
latex=true)

for (i,lbl) in enumerate(keys(profit_dict_3a))

    linestyle=:solid
    color=nothing
    lw=nothing
    if lbl == "Sp-R-IP"
        color = "black"
        lw=llww*1.5
    else
        color = colors[i]
        lw=llww
    end
    plot!([profit_train_pf-profit_dict_3a[lbl][i] for i in 1:length(profit_dict_3a[lbl])],label=lbl,linestyle=linestyle,color=color,linewidth=lw)

end

yaxis!("Regret (€)")
xaxis!("Training iteration")


#Figure 3b
plot(
    xguidefontsize=40,
    yguidefontsize=40,
    xticks=:auto,xtickfontsize=30,
    yticks=:auto,ytickfontsize=30,
    legendfontsize=25,
    size=(1500,1100),
    left_margin = 15Plots.mm,
    bottom_margin= 10Plots.mm,
    latex=true
    )


for (i,lbl) in enumerate(keys(profit_dict_3b))

    linestyle=:solid
    color=nothing
    lw=nothing
    if lbl == "Sp-R-IP"
        color = "black"
        lw=llww*1.5
    else
        color = colors[i]
        lw=llww
    end
    plot!([profit_train_pf-profit_dict_3b[lbl][i] for i in 1:length(profit_dict_3b[lbl])],label=lbl,linestyle=linestyle,color=color,linewidth=lw)

end

yaxis!("Regret (€)")
xaxis!("Training iteration")


#Figure 3c
plot(
    xguidefontsize=40,
    yguidefontsize=40,
    xticks=:auto,xtickfontsize=30,
    yticks=:auto,ytickfontsize=30,
    legendfontsize=25,
    size=(1500,1100),
    left_margin = 15Plots.mm,
    bottom_margin= 10Plots.mm,
    latex=true
    )

for (i,lbl) in enumerate(keys(profit_dict_3c))

    linestyle=:solid
    color=nothing
    lw=nothing
    if lbl == "Sp-R-IP"
        color = "black"
        lw=llww*1.5
    else
        color = colors[i]
        lw=llww
    end
    plot!([profit_val_pf-profit_dict_3c[lbl][i] for i in 1:length(profit_dict_3c[lbl])],label=lbl,linestyle=linestyle,color=color,linewidth=lw)

end

yaxis!("Regret (€)")
xaxis!("Training iteration")


#Figure 3d
plot(
    xguidefontsize=40,
    yguidefontsize=40,
    xticks=:auto,xtickfontsize=30,
    yticks=:auto,ytickfontsize=30,
    legendfontsize=25,
    size=(1500,1100),
    left_margin = 15Plots.mm,
    bottom_margin= 10Plots.mm,
    latex=true
    )

for (i,lbl) in enumerate(keys(profit_dict_3d))

    linestyle=:solid
    color=nothing
    lw=nothing
    if lbl == "Sp-R-IP"
        color = "black"
        lw=llww*1.5
    else
        color = colors[i]
        lw=llww
    end
    plot!([profit_val_pf-profit_dict_3d[lbl][i] for i in 1:length(profit_dict_3d[lbl])],label=lbl,linestyle=linestyle,color=color,linewidth=lw)

end

yaxis!("Regret (€)")
xaxis!("Training iteration")





##### Obtaining Figure 4 #####

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
    "Sp_SG_softplus_warm_fig4",
]


list_evols,list_fc_lists = read_evol_lists(list_dirs,true)
list_evols_red = reduce_evol_lists(list_evols)
fc_price = calc_price_evolution(list_fc_lists,features_train)

index1 = 100
config = 16

evols = list_evols_red[1][config]


c_fc = -fc_price[config][index1][1:24]
c_act = -labels_train[1][1:24]
c_adj = 2 .* c_fc - c_act

w_fc = calc_optimal_schedule(-transpose(c_fc),OP_params_dict,"Gurobi")
w_act = calc_optimal_schedule(-transpose(c_act),OP_params_dict,"Gurobi")
w_adj = calc_optimal_schedule(-transpose(c_adj),OP_params_dict,"Gurobi")

lw = 10


#Figure 4a
p1 = plot(c_fc, label="ĉ",
 linewidth=lw,
 xguidefontsize=40,
 yguidefontsize=40,
 xtickfontsize=26,
 ytickfontsize=26,
 legendfontsize=30,
 legend=:bottomleft,
 size=(1500,800),
 left_margin = 15Plots.mm,
 bottom_margin= 15Plots.mm,
 latex=true)

plot!(c_act, label="c", linewidth=lw)
plot!(c_adj, label="2ĉ -c", linewidth=lw)
xaxis!("Hour of day", fontsize=100)
yaxis!("Cost (€)")
plot!(xticks=(2:2:24, string.(2:2:24))) 

plot!()



#Figure 4b
p2=plot(w_fc[49:end],
label="x*(ĉ)",
linewidth=lw,
xguidefontsize=40,
yguidefontsize=40,
xtickfontsize=26,
ytickfontsize=26,
legendfontsize=30,
legend=:topright,
size=(1500,800),
left_margin = 15Plots.mm,
bottom_margin= 15Plots.mm,
latex=true)

plot!(w_act[49:end],label="x*(c)",linestyle=:dash,linewidth=lw)
plot!(w_adj[49:end],label="x*(2ĉ-c)",linewidth=lw)
xaxis!("Hour of day")
yaxis!("State of Charge (MWh)")
plot!(xticks=(2:2:24, string.(2:2:24)))

plot!()



#Figure 4c
p3= plot(evols["regret_evol_train"],
linewidth=lw,
xguidefontsize=40,
yguidefontsize=40,
xtickfontsize=26,
ytickfontsize=26,
legendfontsize=30,
legend=:topright,
size=(1500,1800),
left_margin = 15Plots.mm,
bottom_margin= 15Plots.mm,
latex=true
)

xaxis!("Epoch")
yaxis!("Train regret (€)")

function scientific_notation(ticks)
    return [@sprintf("%.0e", tick) for tick in ticks]
end

tick_positions = [0.0, 1e-3, 2e-3]
tick_labels = scientific_notation(tick_positions)
yticks!(tick_positions, tick_labels)

plot!(legend=false)