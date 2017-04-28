-- if this causes issues, you may need to:
-- luarocks remove --force ffi
-- and follow instructions to install
-- https://github.com/facebook/luaffifb
local ok, ffi = pcall(require, 'ffi')

local function checkArgument(condition, fn, ud, msg, level)
   local level = level or 3
   if not condition then
      error("bad argument #" .. ud .. " to '" .. fn .. "' (" .. msg .. ")", level)
   end
end

local function checkArgumentType(expected, actual, fn, ud, level)
   local level = level or 3
   if expected ~= actual then
      checkArgument(false, fn, ud, expected .. " expected, got " .. actual, level + 1)
   end
end

if ok then

   local Real2real = {
      Byte='unsigned char',
      Char='char',
      Short='short',
      Int='int',
      Long='long',
      Float='float',
      Double='double',
      Half='THHalf'
   }

   -- Allocator
   ffi.cdef[[
typedef struct THAllocator {
  void* (*malloc)(void*, ptrdiff_t);
  void* (*realloc)(void*, void*, ptrdiff_t);
  void (*free)(void*, void*);
} THAllocator;
]]

   -- Half
   ffi.cdef[[
typedef struct {
  unsigned short x;
} __THHalf;
typedef __THHalf THHalf;
]]

   -- Storage
   for Real, real in pairs(Real2real) do

      local cdefs = [[
typedef struct THRealStorage
{
    real *data;
    ptrdiff_t size;
    int refcount;
    char flag;
    THAllocator *allocator;
    void *allocatorContext;
} THRealStorage;
]]
      cdefs = cdefs:gsub('Real', Real):gsub('real', real)
      ffi.cdef(cdefs)

      local Storage = torch.getmetatable(string.format('torch.%sStorage', Real))
      local Storage_tt = ffi.typeof('TH' .. Real .. 'Storage**')

      rawset(Storage,
             "cdata",
             function(self)
                return Storage_tt(self)[0]
             end)

      rawset(Storage,
             "data",
             function(self)
                return Storage_tt(self)[0].data
             end)
   end

   -- Tensor
   for Real, real in pairs(Real2real) do

      local cdefs = [[
typedef struct THRealTensor
{
    long *size;
    long *stride;
    int nDimension;

    THRealStorage *storage;
    ptrdiff_t storageOffset;
    int refcount;

    char flag;

} THRealTensor;
]]
      cdefs = cdefs:gsub('Real', Real):gsub('real', real)
      ffi.cdef(cdefs)

      local Tensor_type = string.format('torch.%sTensor', Real)
      local Tensor = torch.getmetatable(Tensor_type)
      local Tensor_tt = ffi.typeof('TH' .. Real .. 'Tensor**')

      rawset(Tensor,
             "cdata",
             function(self)
                if not self then return nil; end
                return Tensor_tt(self)[0]
             end)

      rawset(Tensor,
             "data",
             function(self)
                if not self then return nil; end
                self = Tensor_tt(self)[0]
                return self.storage ~= nil and self.storage.data + self.storageOffset or nil
             end)

      -- faster apply (contiguous case)
      if Tensor_type ~= 'torch.HalfTensor' then
         local apply = Tensor.apply
         rawset(Tensor,
                "apply",
                function(self, func)
                   if self:isContiguous() and self.data then
                      local self_d = self:data()
                      for i=0,self:nElement()-1 do
                         local res = func(tonumber(self_d[i])) -- tonumber() required for long...
                         if res then
                            self_d[i] = res
                         end
                      end
                      return self
                   else
                      return apply(self, func)
                   end
                end)

         -- faster map (contiguous case)
         local map = Tensor.map
         rawset(Tensor,
                "map",
                function(self, src, func)
                   checkArgument(torch.isTensor(src), "map", 1, "tensor expected")
                   checkArgumentType(self:type(), src:type(), "map", 1)

                   if self:isContiguous() and src:isContiguous() and self.data and src.data then
                      local self_d = self:data()
                      local src_d = src:data()
                      assert(src:nElement() == self:nElement(), 'size mismatch')
                      for i=0,self:nElement()-1 do
                         local res = func(tonumber(self_d[i]), tonumber(src_d[i])) -- tonumber() required for long...
                         if res then
                            self_d[i] = res
                         end
                      end
                      return self
                   else
                      return map(self, src, func)
                   end
                end)

         -- faster map2 (contiguous case)
         local map2 = Tensor.map2
         rawset(Tensor,
                "map2",
                function(self, src1, src2, func)
                   checkArgument(torch.isTensor(src1), "map", 1, "tensor expected")
                   checkArgument(torch.isTensor(src2), "map", 2, "tensor expected")
                   checkArgumentType(self:type(), src1:type(), "map", 1)
                   checkArgumentType(self:type(), src2:type(), "map", 2)

                   if self:isContiguous() and src1:isContiguous() and src2:isContiguous() and self.data and src1.data and src2.data then
                      local self_d = self:data()
                     local src1_d = src1:data()
                      local src2_d = src2:data()
                      assert(src1:nElement() == self:nElement(), 'size mismatch')
                      assert(src2:nElement() == self:nElement(), 'size mismatch')
                      for i=0,self:nElement()-1 do
                         local res = func(tonumber(self_d[i]), tonumber(src1_d[i]), tonumber(src2_d[i])) -- tonumber() required for long...
                         if res then
                            self_d[i] = res
                         end
                      end
                      return self
                   else
                      return map2(self, src1, src2, func)
                   end
                end)
             end
   end

   -- torch.data
   -- will fail if :data() is not defined
   function torch.data(self, asnumber)
      if not self then return nil; end
      local data = self:data()
      if asnumber then
         return ffi.cast('intptr_t', data)
      else
         return data
      end
   end

   -- torch.cdata
   -- will fail if :cdata() is not defined
   function torch.cdata(self, asnumber)
      if not self then return nil; end
      local cdata = self:cdata()
      if asnumber then
         return ffi.cast('intptr_t', cdata)
      else
         return cdata
      end
   end

end
