require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')

cmd:argument('-model','model checkpoint to use for sampling')

cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',1,' 0 to use max at each timestep, 1 to sample at each timestep')
cmd:option('-primetext',"",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-length',2000,'number of characters to sample')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:option('-output',"result.txt",'file for output')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
opt.model = 'lstm'
opt.temperature = 1
opt.gpuid = 0
opt.verbose = 1

function gprint(str)
    if opt.verbose == 1 then print(str) end
end


if string.len(opt.output) > 0 then file = io.open(opt.output, "a") end

if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        gprint('using OpenCL on GPU ' .. opt.gpuid .. '...')
        gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
        cltorch.setDevice(opt.gpuid + 1) 
        torch.manualSeed(opt.seed)
    else
        gprint('Falling back on CPU mode')
        opt.gpuid = -1 
    end
end

torch.manualSeed(opt.seed)

-- завантаження checkpoint’а
if not lfs.attributes(opt.model, 'mode') then
    gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() -- забезпечення коректної роботи dropout’а

-- ініціалізація словника 
local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

-- ініціалізувати все нулями
local current_state
current_state = {}
for L = 1,checkpoint.opt.num_layers do
    -- c та h для всіх шарів
    local h_init = torch.zeros(1, checkpoint.opt.rnn_size):double()
    h_init = h_init:cl() 
    table.insert(current_state, h_init:clone())
    table.insert(current_state, h_init:clone())
end
state_size = #current_state

-- зробити кілька кроків, базуючись на "seed" тексті (primetext)
local seed_text = opt.primetext
if string.len(seed_text) > 0 then
    gprint('seeding with ' .. seed_text)
    gprint('--------------------------')
    for c in seed_text:gmatch'.' do
        prev_char = torch.Tensor{vocab[c]}
        io.write(ivocab[prev_char[1]])
	if string.len(opt.output) > 0 then file:write(ivocab[prev_char[1]]) end
        prev_char = prev_char:cl()
        local lst = protos.rnn:forward{prev_char, unpack(current_state)}

        current_state = {}
        for i=1,state_size do table.insert(current_state, lst[i]) end
        prediction = lst[#lst] -- last element holds the log probabilities
    end
else
    -- інакше імовірності кожного елементу зі словника однакові
    gprint('missing seed text, using uniform probability over first character')
    gprint('--------------------------')
    prediction = torch.Tensor(1, #ivocab):fill(1)/(#ivocab)
    if opt.gpuid >= 0 and opt.opencl == 1 then prediction = prediction:cl() end
end

-- початок семплування
for i=1, opt.length do

    -- отримання ймовірностей з попереднього кроку
    if opt.sample == 0 then
        -- обрати максимальну
        local _, prev_char_ = prediction:max(2)
        prev_char = prev_char_:resize(1)
    else
        -- застосувати семплування
        prediction:div(opt.temperature) -- масштабувати по температурі
        local probs = torch.exp(prediction):squeeze()
        probs:div(torch.sum(probs)) -- перенормувати суму probs до одиниці
        prev_char = torch.multinomial(probs:float(), 1):resize(1):float()
    end

    -- наступний крок вперед
    local lst = protos.rnn:forward{prev_char, unpack(current_state)}
    current_state = {}
    for i=1,state_size do table.insert(current_state, lst[i]) end
    prediction = lst[#lst] -- в останньому елеметні інформація про ймовірності
    
    io.write(ivocab[prev_char[1]])
    if string.len(opt.output) > 0 then file:write(ivocab[prev_char[1]]) end
end
io.write('\n') io.flush()

if string.len(opt.output) > 0 then file:write('\n') end
file:close() 
