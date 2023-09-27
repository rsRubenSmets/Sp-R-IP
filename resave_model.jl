using Flux
using BSON: @load

dir_fc = "C:/Users/u0137781/OneDrive - KU Leuven/da_scheduling/Data/pretrained_fc/"
model_save_code = "model_softplus_wd_nlfr_genfc_yhat_scaled_pos.bson"
# Load the model from the BSON file
@load dir_fc*model_save_code model_extended

save_loc = dir_fc*model_save_code[1:end-5]

# Save weights and biases
for (i, layer) in enumerate(model_extended.layers)
    if isa(layer, Dense)
        # Save weights
        weights_filename = "/weights_layer_$(i).npz"
        NPZ.npzwrite(save_loc*weights_filename, Array(layer.weight))
        
        # Save biases
        biases_filename = "/biases_layer_$(i).npz"
        NPZ.npzwrite(save_loc*biases_filename, Array(layer.bias))
    end
end

# Save them to CSV or another format...