
local LSTM = {}
function LSTM.lstm(input_size, rnn_size, n, dropout)
  dropout = dropout or 0 

  -- 2*n+1
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h з попередніх кроків
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- input до цього шару 
    if L == 1 then 
      x = OneHot(input_size)(inputs[1])
      input_size_L = input_size
    else 
      x = outputs[(L-1)*2] 
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- застосувати dropout, якщо він вказаний
      input_size_L = rnn_size
    end
    -- оцінка ефективності вхідних даних (сума двох)
    local i2h = nn.Linear(input_size_L, 4 * rnn_size)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(rnn_size, 4 * rnn_size)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})
    -- розбити вхідні дані (матрицю) на чотири однакові вектори
    local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- поставити "ворота" сигмоїдної фільтрації
    local forget_gate = nn.Sigmoid()(n1)
    local in_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- обробка входу гіп. тангенсом
    local in_transform = nn.Tanh()(n4)
    -- оновити "пам’ять" LSTM клітини (Cn -> Cn+1)
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- сформувати вихідний вектор клітини (h)
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})
    
    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  return nn.gModule(inputs, outputs)
end

return LSTM
