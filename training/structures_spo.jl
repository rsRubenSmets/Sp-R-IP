
module NNStructure

    using Random

    struct NeuralNetwork
        weights::Vector{Array{Float64,2}}  # Vector of weight matrices for each layer
        biases::Vector{Vector{Float64}}     # Vector of bias vectors for each layer
        activations::Vector{Any}              # Array of activation functions for each layer



        function NeuralNetwork(weights::Vector{Array{Float64,2}}, biases::Vector{Vector{Float64}}, activations_str::Vector{String})
            #Initialization by explicitly giving weights and bias vectors
            activations = convert_vector_str_to_fct(activations_str)

            new(weights, biases, activations)
        end

        function NeuralNetwork(layer_sizes::Vector{Int}, activations_str::Vector{String},rd_seed::Int=0)
            #Random initialization given the size of the layers and activation functions
            activations = convert_vector_str_to_fct(activations_str)

            if rd_seed>0
                Random.seed!(rd_seed)
            end

            # Initialize weights and biases
            weights = [randn(layer_sizes[i+1], layer_sizes[i]) for i in 1:length(layer_sizes)-1]
            biases = [randn(layer_sizes[i]) for i in 2:length(layer_sizes)]

            new(weights, biases, activations)
        end

    end

    function (nn::NeuralNetwork)(x::Vector{Float64})
        for (i, (W, b)) in enumerate(zip(nn.weights, nn.biases))
            x = nn.activations[i](W * x .+ b)
        end
        return x
    end

    function convert_vector_str_to_fct(activation_strs::Vector{String})

        activations = []
            
        for act_str in activation_strs
            act_fct = convert_act_str_fct_single(act_str)  # Assuming this function is defined elsewhere
            push!(activations, act_fct)
        end

        return activations

    end

    function convert_act_str_fct_single(str)
        if str == "relu"
            return relu
        elseif str == "softplus"
            return softplus
        else
            error("Unsupported activaton function $(str)")
        end
    end

    function relu(x)
        return max.(0,x)
    end

    function softplus(x)
        return log.(1 .+ exp.(x))
    end

end

module TrainingOutputStructure
    x=1
end


using .NNStructure
using Random

layer_sizes = [5, 10, 3]
act_strs = ["relu","softplus"]
rd_seed = 0

Random.seed!(rd_seed)   

weights = [randn(layer_sizes[i+1], layer_sizes[i]) for i in 1:length(layer_sizes)-1]
biases = [randn(layer_sizes[i]) for i in 2:length(layer_sizes)]


# Creating a NeuralNetwork instance
nn_expl = NNStructure.NeuralNetwork(weights,biases,act_strs)
nn_rd = NNStructure.NeuralNetwork(layer_sizes,act_strs,rd_seed)

# Example input
input_features = randn(5)

# Calling the neural network with the input
output_expl = nn_expl(input_features)
output_rd = nn_rd(input_features)


@show(output_expl)

@show(output_rd)


x=1