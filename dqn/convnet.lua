--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "initenv"
require './BinaryLinear.lua'
require './BinarizedNeurons'
require 'cudnn'
require './cudnnBinarySpatialConvolution.lua'
require './BatchNormalizationShiftPow2.lua'
require './SpatialBatchNormalizationShiftPow2.lua'


function create_network(args)

    local BatchNormalization
    local SpatialBatchNormalization
    BatchNormalization = BatchNormalizationShiftPow2
    SpatialBatchNormalization = SpatialBatchNormalizationShiftPow2

    --local SpatialConvolution
    --SpatialConvolution = cudnnBinarySpatialConvolution

    local net = nn.Sequential()
    net:add(nn.Reshape(unpack(args.input_dims)))

    --- first convolutional layer
    local convLayer = cudnnBinarySpatialConvolution

    net:add(convLayer(args.hist_len*args.ncols, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],1,1,false))-- stcWieghts = false
    net:add(SpatialBatchNormalization(64, true)) -- valRunning = true
    net:add(args.nl())
    net:add(BinarizedNeurons(true)) -- stcNeurons = true
    
    -- Add convolutional layers
    for i=1,(#args.n_units-1) do
        -- second convolutional layer
        net:add(convLayer(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1], false)) -- stcWieghts = false
        net:add(SpatialBatchNormalization(64, true)) -- valRunning = true
        net:add(args.nl())
        net:add(BinarizedNeurons(true)) -- stcNeurons = true
    end

    local nel
    if args.gpu >= 0 then
        nel = net:cuda():forward(torch.zeros(1,unpack(args.input_dims))
                :cuda()):nElement()
    else
        nel = net:forward(torch.zeros(1,unpack(args.input_dims))):nElement()
    end

    -- reshape all feature planes into a vector per example
    net:add(nn.Reshape(nel))

    -- fully connected layer
    net:add(BinaryLinear(nel, args.n_hid[1], false)) -- stcWeights = false
    net:add(BatchNormalization(args.n_hid[1], true))-- runningVal = true
    net:add(args.nl())
    net:add(BinarizedNeurons(true)) -- stcNeurons = true

    local last_layer_size = args.n_hid[1]

    for i=1,(#args.n_hid-1) do
        -- add Linear layer
        last_layer_size = args.n_hid[i+1]
        net:add(nn.Linear(args.n_hid[i], last_layer_size, false)) -- stcWeights
        net:add(BatchNormalization(last_layer_size, true))-- runningVal = true
        net:add(args.nl())
        net:add(BinarizedNeurons(true)) -- stcNeurons = true
    end

    -- add the last fully connected layer (to actions)
    net:add(BinaryLinear(last_layer_size, args.n_actions,false)) --stcWeights
    net:add(nn.BatchNormalization(args.n_actions))

    if args.gpu >=0 then
        net:cuda()
    end
    if args.verbose >= 2 then
        print(net)
        print('Convolutional layers flattened output size:', nel)
    end
    return net
end
