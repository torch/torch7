-- additional methods for Storage
local Storage = {}

-- additional methods for Tensor
local Tensor = {}

-- types
local types = {'Byte', 'Char', 'Short', 'Int', 'Long', 'Float', 'Double'}

-- tostring() functions for Tensor and Storage
local function Storage__printformat(self)
   if self:size() == 0 then 
     return "", nil, 0
   end
   local intMode = true
   local type = torch.typename(self)
--   if type == 'torch.FloatStorage' or type == 'torch.DoubleStorage' then
      for i=1,self:size() do
         if self[i] ~= math.ceil(self[i]) then
            intMode = false
            break
         end
      end
--   end
   local tensor = torch.DoubleTensor(torch.DoubleStorage(self:size()):copy(self), 1, self:size()):abs()
   local expMin = tensor:min()
   if expMin ~= 0 then
      expMin = math.floor(math.log10(expMin)) + 1
   else
      expMin = 1
   end
   local expMax = tensor:max()
   if expMax ~= 0 then
      expMax = math.floor(math.log10(expMax)) + 1
   else
      expMax = 1
   end

   local format
   local scale
   local sz
   if intMode then
      if expMax > 9 then
         format = "%11.4e"
         sz = 11
      else
         format = "%SZd"
         sz = expMax + 1
      end
   else
      if expMax-expMin > 4 then
         format = "%SZ.4e"
         sz = 11
         if math.abs(expMax) > 99 or math.abs(expMin) > 99 then
            sz = sz + 1
         end
      else
         if expMax > 5 or expMax < 0 then
            format = "%SZ.4f"
            sz = 7
            scale = math.pow(10, expMax-1)
         else
            format = "%SZ.4f"
            if expMax == 0 then
               sz = 7
            else
               sz = expMax+6
            end
         end
      end
   end
   format = string.gsub(format, 'SZ', sz)
   if scale == 1 then
      scale = nil
   end
   return format, scale, sz
end

function Storage.__tostring__(self)
   local strt = {'\n'}
   local format,scale = Storage__printformat(self)
   if format:sub(2,4) == 'nan' then format = '%f' end
   if scale then
      table.insert(strt, string.format('%g', scale) .. ' *\n')
      for i = 1,self:size() do
         table.insert(strt, string.format(format, self[i]/scale) .. '\n')
      end
   else
      for i = 1,self:size() do
         table.insert(strt, string.format(format, self[i]) .. '\n')
      end
   end
   table.insert(strt, '[' .. torch.typename(self) .. ' of size ' .. self:size() .. ']\n')
   local str = table.concat(strt)
   return str
end

for _,type in ipairs(types) do
   local metatable = torch.getmetatable('torch.' .. type .. 'Storage')
   for funcname, func in pairs(Storage) do
      rawset(metatable, funcname, func)
   end
end

local function Tensor__printMatrix(self, indent)
   local format,scale,sz = Storage__printformat(self:storage())
   if format:sub(2,4) == 'nan' then format = '%f' end
--   print('format = ' .. format)
   scale = scale or 1
   indent = indent or ''
   local strt = {indent}
   local nColumnPerLine = math.floor((80-#indent)/(sz+1))
--   print('sz = ' .. sz .. ' and nColumnPerLine = ' .. nColumnPerLine)
   local firstColumn = 1
   local lastColumn = -1
   while firstColumn <= self:size(2) do
      if firstColumn + nColumnPerLine - 1 <= self:size(2) then
         lastColumn = firstColumn + nColumnPerLine - 1
      else
         lastColumn = self:size(2)
      end
      if nColumnPerLine < self:size(2) then
         if firstColumn ~= 1 then
            table.insert(strt, '\n')
         end
         table.insert(strt, 'Columns ' .. firstColumn .. ' to ' .. lastColumn .. '\n' .. indent)
      end
      if scale ~= 1 then
         table.insert(strt, string.format('%g', scale) .. ' *\n ' .. indent)
      end
      for l=1,self:size(1) do
         local row = self:select(1, l)
         for c=firstColumn,lastColumn do
            table.insert(strt, string.format(format, row[c]/scale))
            if c == lastColumn then
               table.insert(strt, '\n')
               if l~=self:size(1) then
                  if scale ~= 1 then
                     table.insert(strt, indent .. ' ')
                  else
                     table.insert(strt, indent)
                  end
               end
            else
               table.insert(strt, ' ')
            end
         end
      end
      firstColumn = lastColumn + 1
   end
   local str = table.concat(strt)
   return str
end

local function Tensor__printTensor(self)
   local counter = torch.LongStorage(self:nDimension()-2)
   local strt = {''}
   local finished
   counter:fill(1)
   counter[1] = 0
   while true do
      for i=1,self:nDimension()-2 do
         counter[i] = counter[i] + 1
         if counter[i] > self:size(i) then
            if i == self:nDimension()-2 then
               finished = true
               break
            end
            counter[i] = 1
         else
            break
         end
      end
      if finished then
         break
      end
--      print(counter)
      if #strt > 1 then
         table.insert(strt, '\n')
      end
      table.insert(strt, '(')
      local tensor = self
      for i=1,self:nDimension()-2 do
         tensor = tensor:select(1, counter[i])
         table.insert(strt, counter[i] .. ',')
      end
      table.insert(strt, '.,.) = \n')
      table.insert(strt, Tensor__printMatrix(tensor, ' '))
   end
   local str = table.concat(strt)
   return str
end

function Tensor.__tostring__(self)
   local str = '\n'
   local strt = {''}
   if self:nDimension() == 0 then
      table.insert(strt, '[' .. torch.typename(self) .. ' with no dimension]\n')
   else
      local tensor = torch.DoubleTensor():resize(self:size()):copy(self)
      if tensor:nDimension() == 1 then
         local format,scale,sz = Storage__printformat(tensor:storage())
	 if format:sub(2,4) == 'nan' then format = '%f' end
         if scale then
            table.insert(strt, string.format('%g', scale) .. ' *\n')
            for i = 1,tensor:size(1) do
               table.insert(strt, string.format(format, tensor[i]/scale) .. '\n')
            end
         else
            for i = 1,tensor:size(1) do
               table.insert(strt, string.format(format, tensor[i]) .. '\n')
            end
         end
         table.insert(strt, '[' .. torch.typename(self) .. ' of size ' .. tensor:size(1) .. ']\n')
      elseif tensor:nDimension() == 2 then
         table.insert(strt, Tensor__printMatrix(tensor))
         table.insert(strt, '[' .. torch.typename(self) .. ' of size ' .. tensor:size(1) .. 'x' .. tensor:size(2) .. ']\n')
      else
         table.insert(strt, Tensor__printTensor(tensor))
         table.insert(strt, '[' .. torch.typename(self) .. ' of size ')
         for i=1,tensor:nDimension() do
            table.insert(strt, tensor:size(i))
            if i ~= tensor:nDimension() then
               table.insert(strt, 'x')
            end
         end
         table.insert(strt, ']\n')
      end
   end
   local str = table.concat(strt)
   return str
end

function Tensor.type(self,type)
   local current = torch.typename(self)
   if not type then return current end
   if type ~= current then
      local new = torch.getmetatable(type).new()
      if self:nElement() > 0 then
         new:resize(self:size()):copy(self)
      end
      return new
   else
      return self
   end
end

function Tensor.typeAs(self,tensor)
   return self:type(tensor:type())
end

function Tensor.byte(self)
   return self:type('torch.ByteTensor')
end

function Tensor.char(self)
   return self:type('torch.CharTensor')
end

function Tensor.short(self)
   return self:type('torch.ShortTensor')
end

function Tensor.int(self)
   return self:type('torch.IntTensor')
end

function Tensor.long(self)
   return self:type('torch.LongTensor')
end

function Tensor.float(self)
   return self:type('torch.FloatTensor')
end

function Tensor.double(self)
   return self:type('torch.DoubleTensor')
end

function Tensor.real(self)
   return self:type(torch.getdefaulttensortype())
end

function Tensor.expand(result,tensor,...)
   -- get sizes
   local sizes = {...}

   local t = torch.type(tensor)
   if (t == 'number' or t == 'torch.LongStorage') then
      table.insert(sizes,1,tensor)
      tensor = result
      result = tensor.new()
   end

   -- check type
   local size
   if torch.type(sizes[1])=='torch.LongStorage' then
      size = sizes[1]
   else
      size = torch.LongStorage(#sizes)
      for i,s in ipairs(sizes) do
         size[i] = s
      end
   end

   -- get dimensions
   local tensor_dim = tensor:dim()
   local tensor_stride = tensor:stride()
   local tensor_size = tensor:size()

   -- check nb of dimensions
   if #size ~= tensor:dim() then
      error('the number of dimensions provided must equal tensor:dim()')
   end

   -- create a new geometry for tensor:
   for i = 1,tensor_dim do
      if tensor_size[i] == 1 then
         tensor_size[i] = size[i]
         tensor_stride[i] = 0
      elseif tensor_size[i] ~= size[i] then
         error('incorrect size: only supporting singleton expansion (size=1)')
      end
   end

   -- create new view, with singleton expansion:
   result:set(tensor:storage(), tensor:storageOffset(),
                         tensor_size, tensor_stride)
   return result
end
torch.expand = Tensor.expand

function Tensor.expandAs(result,tensor,template)
   if template then
      return result:expand(tensor,template:size())
   end
   return result:expand(tensor:size())
end
torch.expandAs = Tensor.expandAs

function Tensor.repeatTensor(result,tensor,...)
   -- get sizes
   local sizes = {...}

   local t = torch.type(tensor)
   if (t == 'number' or t == 'torch.LongStorage') then
      table.insert(sizes,1,tensor)
      tensor = result
      result = tensor.new()
   end
   -- if not contiguous, then force the tensor to be contiguous
   if not tensor:isContiguous() then tensor = tensor:clone() end

   -- check type
   local size
   if torch.type(sizes[1])=='torch.LongStorage' then
      size = sizes[1]
   else
      size = torch.LongStorage(#sizes)
      for i,s in ipairs(sizes) do
         size[i] = s
      end
   end
   if size:size() < tensor:dim() then
      error('Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor')
   end
   local xtensor = tensor.new():set(tensor)
   local xsize = xtensor:size():totable()
   for i=1,size:size()-tensor:dim() do
      table.insert(xsize,1,1)
   end
   size = torch.DoubleTensor(xsize):cmul(torch.DoubleTensor(size:totable())):long():storage()
   xtensor:resize(torch.LongStorage(xsize))
   result:resize(size)
   local urtensor = result.new(result)
   for i=1,xtensor:dim() do
      urtensor = urtensor:unfold(i,xtensor:size(i),xtensor:size(i))
   end
   for i=1,urtensor:dim()-xtensor:dim() do
      table.insert(xsize,1,1)
   end
   xtensor:resize(torch.LongStorage(xsize))
   local xxtensor = xtensor:expandAs(urtensor)
   urtensor:copy(xxtensor)
   return result
end
torch.repeatTensor = Tensor.repeatTensor

--- One of the size elements can be -1,
 --- a new LongStorage is then returned.
 --- The length of the unspecified dimension
 --- is infered from the number of remaining elements.
local function specifyFully(size, nElements)
    local nCoveredElements = 1
    local remainingDim = nil
    local sizes = size:totable()
    for i = 1, #sizes do
        local wantedDimSize = sizes[i]
        if wantedDimSize == -1 then
            if remainingDim then
                error("Only one of torch.view dimensions can be -1.")
            end
            remainingDim = i
        else
            nCoveredElements = nCoveredElements * wantedDimSize
        end
    end

    if not remainingDim then
        return size
    end

    assert(nElements % nCoveredElements == 0, "The number of covered elements is not a multiple of all elements.")
    local copy = torch.LongStorage(sizes)
    copy[remainingDim] = nElements / nCoveredElements
    return copy
end

-- TODO : This should be implemented in TH and and wrapped.
function Tensor.view(result, src, ...)
   local size = ...
   local view, tensor
   local function istensor(tensor)
      return torch.typename(tensor) and torch.typename(tensor):find('torch.*Tensor')
   end
   local function isstorage(storage)
      return torch.typename(storage) and torch.typename(storage) == 'torch.LongStorage'
   end
   if istensor(result) and istensor(src) and type(size) == 'number' then
      size = torch.LongStorage{...}
      view = result
      tensor = src
   elseif istensor(result) and istensor(src) and isstorage(size) then
      size = size
      view = result
      tensor = src
   elseif istensor(result) and isstorage(src) and size == nil then
      size = src
      tensor = result
      view = tensor.new()
   elseif istensor(result) and type(src) == 'number' then
      size = {...}
      table.insert(size,1,src)
      size = torch.LongStorage(size)
      tensor = result
      view = tensor.new()
   else
      local t1 = 'torch.Tensor, torch.Tensor, number [, number ]*'
      local t2 = 'torch.Tensor, torch.Tensor, torch.LongStorage'
      local t3 = 'torch.Tensor, torch.LongStorage'
      local t4 = 'torch.Tensor, number [, number ]*'
      error(string.format('torch.view, expected (%s) or\n (%s) or\n (%s)\n or (%s)', t1, t2, t3, t4))
   end
   local origNElement = tensor:nElement()
   size = specifyFully(size, origNElement)

   assert(tensor:isContiguous(), "expecting a contiguous tensor")
   view:set(tensor:storage(), tensor:storageOffset(), size)
   if view:nElement() ~= origNElement then
      local inputSize = table.concat(tensor:size():totable(), "x")
      local outputSize = table.concat(size:totable(), "x")
      error(string.format("Wrong size for view. Input size: %s. Output size: %s",
      inputSize, outputSize))
   end
   return view
end
torch.view = Tensor.view

function Tensor.viewAs(result, src, template)
   if template and torch.typename(template) then
      return result:view(src, template:size())
   elseif template == nil then
      template = src
      src = result
      result = src.new()
      return result:view(src, template:size())
   else
      local t1 = 'torch.Tensor, torch.Tensor, torch.LongStorage'
      local t2 = 'torch.Tensor, torch.LongStorage'
      error(string.format('expecting (%s) or (%s)', t1, t2))
   end
end
torch.viewAs = Tensor.viewAs

function Tensor.split(result, tensor, splitSize, dim)
   if torch.type(result) ~= 'table' then
      dim = splitSize
      splitSize = tensor
      tensor = result
      result = {}
   else
      -- empty existing result table before using it
      for k,v in pairs(result) do
         result[k] = nil
      end
   end
   dim = dim or 1
   local start = 1
   while start <= tensor:size(dim) do
      local size = math.min(splitSize, tensor:size(dim) - start + 1)
      local split = tensor:narrow(dim, start, size)
      table.insert(result, split)
      start = start + size
   end
   return result
end
torch.split = Tensor.split

function Tensor.chunk(result, tensor, nChunk, dim)
   if torch.type(result) ~= 'table' then
      dim = nChunk
      nChunk = tensor
      tensor = result
      result = {}
   end
   dim = dim or 1
   local splitSize = math.ceil(tensor:size(dim)/nChunk)
   return torch.split(result, tensor, splitSize, dim)
end
torch.chunk = Tensor.chunk

function Tensor.totable(tensor)
  local result = {}
  if tensor:dim() == 1 then
    tensor:apply(function(i) table.insert(result, i) end)
  else
    for i = 1, tensor:size(1) do
      table.insert(result, tensor[i]:totable())
    end
  end
  return result
end
torch.totable = Tensor.totable

function Tensor.permute(tensor, ...)
  local perm = {...}
  local nDims = tensor:dim()
  assert(#perm == nDims, 'Invalid permutation')
  local j
  for i, p in ipairs(perm) do
    if p ~= i and p ~= 0 then
      j = i
      repeat
        assert(0 < perm[j] and perm[j] <= nDims, 'Invalid permutation')
        tensor = tensor:transpose(j, perm[j])
        j, perm[j] = perm[j], 0
      until perm[j] == i
      perm[j] = j
    end
  end
  return tensor
end
torch.permute = Tensor.permute

for _,type in ipairs(types) do
   local metatable = torch.getmetatable('torch.' .. type .. 'Tensor')
   for funcname, func in pairs(Tensor) do
      rawset(metatable, funcname, func)
   end
end
