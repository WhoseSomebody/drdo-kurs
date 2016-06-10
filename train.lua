require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'
local CharSplitLMMinibatchLoader = require 'util.CharSplitLMMinibatchLoader'
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')

cmd:option('-data_dir','data/tinyshakespeare','data directory. Should contain the file input.txt with input data')

cmd:option('-rnn_size', 128, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-opencl',0,'use OpenCL')
cmd:text()


opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
opt.max_epochs = 50 -- максимальна кількість епох
opt.model = 'lstm'
opt.learning_rate = 2e-3 
opt.learning_rate_decay_after = 10 -- ослаблення частоти через *** епох
opt.decay_rate = 0.95 -- сила послаблення
opt.dropout = 0
opt.seq_length = 50 -- коейіцієнт відрізку
opt.batch_size = 50 -- коефіцієнт порції
opt.grad_clip = 5 -- обмеження градієнту
opt.train_frac = 0.95 -- процент інформації для тренування
opt.val_frac = 0.05 -- процент інформації лдя валідації
opt.init_from =  ''
opt.print_every = 1
opt.eval_val_every = 1000 -- зберігати результат кожні *** ітерацій
opt.checkpoint_dir = 'cv'
opt.savefile = 'lstm'
opt.accurate_gpu_timing = 0
opt.gpuid = 0

-- розділити вхідні дані на train / val / test  (навчання, валідація, тестування)
local test_frac = math.max(0, 1 - (opt.train_frac + opt.val_frac))
local split_sizes = {opt.train_frac, opt.val_frac, test_frac} 

if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        print('using OpenCL on GPU ' .. opt.gpuid .. '...')
        cltorch.setDevice(opt.gpuid + 1)
        torch.manualSeed(opt.seed)
    else
        print('Falling back on CPU mode')
        opt.gpuid = -1 
    end
end

-- об’єкт завантаження вхідних даних
local loader = CharSplitLMMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes)
local vocab_size = loader.vocab_size  -- кількість унікальних символів (словник)
local vocab = loader.vocab_mapping
print('vocab size: ' .. vocab_size)

if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- визначення стану поточної моделі (кількість шарів, розмір мережі, словника та дроп-аут)
local do_random_init = true
if string.len(opt.init_from) > 0 then
    print('loading a model from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
    -- перевірка збіжності словників
    local vocab_compatible = true
    local checkpoint_vocab_size = 0
    for c,i in pairs(checkpoint.vocab) do
        if not (vocab[c] == i) then
            vocab_compatible = false
        end
        checkpoint_vocab_size = checkpoint_vocab_size + 1
    end
    if not (checkpoint_vocab_size == vocab_size) then
        vocab_compatible = false
        print('checkpoint_vocab_size: ' .. checkpoint_vocab_size)
    end
    assert(vocab_compatible, 'error, the character vocabulary for this dataset and the one in the saved checkpoint are not the same. This is trouble.')
    -- перезапис параметрів моделі, що базуються на checkpoint для забезпеченя сходженняя
    print('overwriting rnn_size=' .. checkpoint.opt.rnn_size .. ', num_layers=' .. checkpoint.opt.num_layers .. ', model=' .. checkpoint.opt.model .. ' based on the checkpoint.')
    opt.rnn_size = checkpoint.opt.rnn_size
    opt.num_layers = checkpoint.opt.num_layers
    opt.model = checkpoint.opt.model
    do_random_init = false
else
    print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
    protos = {}
    protos.rnn = LSTM.lstm(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
    protos.criterion = nn.ClassNLLCriterion()
end



-- початкова ініціалізація клітини LSTM мережі
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    h_init = h_init:cl()
    table.insert(init_state, h_init:clone())
end

for k,v in pairs(protos) do v:cl() end


-- запис всіх вище встановлених параметрів у один великий вектор (сама мережа ваг, по суті)
params, grad_params = model_utils.combine_all_parameters(protos.rnn)

-- випадкова ініціалізація
if do_random_init then
    params:uniform(-0.08, 0.08) 
end
-- ініціалізувати forget gates з відносно великими (=1) біасами, щоб заохотити запам’ятовування на початку навчання
for layer_idx = 1, opt.num_layers do
    for _,node in ipairs(protos.rnn.forwardnodes) do
        if node.data.annotations.name == "i2h_" .. layer_idx then
            print('setting forget gate biases to 1 in LSTM layer ' .. layer_idx)
            node.data.module.bias[{{opt.rnn_size+1, 2*opt.rnn_size}}]:fill(1.0)
        end
    end
end


print('number of parameters in the model: ' .. params:nElement())
-- зробити багато клонів, щоб перерозподілити пам’ять (ОЛ)
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- попередня обробка/оптимізація даних
function prepro(x,y)
    x = x:transpose(1,2):contiguous() -- транспонувати вектори для швидшої обробки (ОЛ)
    y = y:transpose(1,2):contiguous()
    x = x:cl()
    y = y:cl()
    return x,y
end

-- оцінка помилки на інтервалі 
function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_index) -- перемістити покажчика ітератора цеї порції інформації наперед відрізка
    local loss = 0
    local rnn_state = {[0] = init_state}
    
    for i = 1,n do -- цикл по порціях на даному відрізку 
        -- взяти одну порцію
        local x, y = loader:next_batch(split_index)
        x,y = prepro(x,y)
        -- "forward pass"
        for t=1,opt.seq_length do
            clones.rnn[t]:evaluate() -- необхідна оцінка для коректної роботи dropout’а
            local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}
            rnn_state[t] = {}
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
            prediction = lst[#lst] 
            loss = loss + clones.criterion[t]:forward(prediction, y[t])
        end
        -- перенесення стану мережі (n->0)
        rnn_state[0] = rnn_state[#rnn_state]
        print(i .. '/' .. n .. '...')
    end
    -- середня помилка на всьому відрізку
    loss = loss / opt.seq_length / n
    return loss
end

-- fwd/bwd прогін (результат - помилка та параметр градієнта)
local init_state_global = clone_list(init_state)
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ---- відрізати маленьку порцію інформації
    local x, y = loader:next_batch(1)
    x,y = prepro(x,y)
    --------- (fwd)
    local rnn_state = {[0] = init_state_global}
    local loss = 0
    for t=1,opt.seq_length do
        clones.rnn[t]:training() 
        local lst = clones.rnn[t]:forward{x[t], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- витягування лише векторів стану ("пам’яті") 
        predictions[t] = lst[#lst] -- останній елемент - орієнтир (здогадка)
        loss = loss + clones.criterion[t]:forward(predictions[t], y[t])
    end
    loss = loss / opt.seq_length

    --------- (bwd)
    -- ініціалізувати градієнт 0-ми (з "майбутнього" впливу немає)
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} 
    for t=opt.seq_length,1,-1 do
        -- зворотнє поширення помилки
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[t])
        table.insert(drnn_state[t], doutput_t)
        local dlst = clones.rnn[t]:backward({x[t], unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- якщо k == 1 то градієнт на x 
                drnn_state[t-1][k-1] = v
            end
        end
    end

    -- передати кінцевий стан в початковий (BPTT)
    init_state_global = rnn_state[#rnn_state] 
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end

-- власне, оптимізація 
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil
for i = 1, iterations do
    local epoch = i / loader.ntrain
    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval, params, optim_state)
    local time = timer:time().real
    
    local train_loss = loss[1] -- помилка знаходиться в списку посередині
    train_losses[i] = train_loss

    -- ослаблення дуже стрімкого learning rate
    if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- ослаблення
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    -- запис в чекпоінт
    if i % opt.eval_val_every == 0 or i == iterations then
        -- оцінка помилки на валідаціній інформації
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[i] = val_loss

        local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end
   
    if i % 10 == 0 then collectgarbage() end

    -- якщо щось пішло не так, закінчити виконання
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break 
    end
    if loss0 == nil then loss0 = loss[1] end
    if loss[1] > loss0 * 3 then
        print('loss is exploding, aborting.')
        break 
    end
end


