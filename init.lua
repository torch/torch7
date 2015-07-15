
-- We are using paths.require to appease mkl

-- Make this work with LuaJIT in Lua 5.2 compatibility mode, which
-- renames string.gfind (already deprecated in 5.1)
if not string.gfind then
    string.gfind = string.gmatch
end
if not table.unpack then
    table.unpack = unpack
end

require "paths"
paths.require "libtorch"

--- package stuff
function torch.packageLuaPath(name)
   if not name then
      local ret = string.match(torch.packageLuaPath('torch'), '(.*)/')
       if not ret then --windows?
           ret = string.match(torch.packageLuaPath('torch'), '(.*)\\')
       end
       return ret
   end
   for path in string.gmatch(package.path, "[^;]+") do
      path = string.gsub(path, "%?", name)
      local f = io.open(path)
      if f then
         f:close()
         local ret = string.match(path, "(.*)/")
         if not ret then --windows?
             ret = string.match(path, "(.*)\\")
         end
         return ret
      end
   end
end

local function include(file, depth)
   paths.dofile(file, 3 + (depth or 0))
end
rawset(_G, 'include', include)

function torch.include(package, file)
   dofile(torch.packageLuaPath(package) .. '/' .. file)
end

function torch.class(tname, parenttname)

   local function constructor(...)
      local self = {}
      torch.setmetatable(self, tname)
      if self.__init then
         self:__init(...)
      end
      return self
   end

   local function factory()
      local self = {}
      torch.setmetatable(self, tname)
      return self
   end

   local mt = torch.newmetatable(tname, parenttname, constructor, nil, factory)
   local mpt
   if parenttname then
      mpt = torch.getmetatable(parenttname)
   end
   return mt, mpt
end

function torch.setdefaulttensortype(typename)
   assert(type(typename) == 'string', 'string expected')
   if torch.getconstructortable(typename) then
      torch.Tensor = torch.getconstructortable(typename)
      torch.Storage = torch.getconstructortable(torch.typename(torch.Tensor(1):storage()))
   else
      error(string.format("<%s> is not a string describing a torch object", typename))
   end
end

function torch.type(obj)
   local class = torch.typename(obj)
   if not class then
      class = type(obj)
   end
   return class
end

--[[ See if a given object is an instance of the provided torch class. ]]
function torch.isTypeOf(obj, typeSpec)
  -- typeSpec can be provided as either a string, regexp, or the constructor. If
  -- the constructor is used, we look in the __typename field of the
  -- metatable to find a string to compare to.
  if type(typeSpec) ~= 'string' then
    typeSpec = getmetatable(typeSpec).__typename
    assert(type(typeSpec) == 'string',
           "type must be provided as [regexp] string, or factory")
  end

  local mt = getmetatable(obj)
  while mt do
    if mt.__typename and mt.__typename:find(typeSpec) then
      return true
    end
    mt = getmetatable(mt)
  end
  return false
end

torch.setdefaulttensortype('torch.DoubleTensor')

include('Tensor.lua')
include('File.lua')
include('CmdLine.lua')
include('FFI.lua')
include('Tester.lua')
include('test.lua')

function torch.totable(obj)
  if torch.isTensor(obj) or torch.isStorage(obj) then
    return obj:totable()
  else
    error("obj must be a Storage or a Tensor")
  end
end

function torch.isTensor(obj)
   local typename = torch.typename(obj)
   if typename and typename:find('torch.*Tensor') then
      return true
   end
   return false
end

function torch.isStorage(obj)
   local typename = torch.typename(obj)
   if typename and typename:find('torch.*Storage') then
      return true
   end
   return false
end
-- alias for convenience
torch.Tensor.isTensor = torch.isTensor

return torch
