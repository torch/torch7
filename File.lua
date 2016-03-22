local File = torch.getmetatable('torch.File')

function File:writeBool(value)
   if value then
      self:writeInt(1)
   else
      self:writeInt(0)
   end
end

function File:readBool()
   return (self:readInt() == 1)
end

local TYPE_NIL      = 0
local TYPE_NUMBER   = 1
local TYPE_STRING   = 2
local TYPE_TABLE    = 3
local TYPE_TORCH    = 4
local TYPE_BOOLEAN  = 5
local TYPE_FUNCTION = 6
local TYPE_RECUR_FUNCTION = 8
local LEGACY_TYPE_RECUR_FUNCTION = 7

-- Lua 5.2 compatibility
local loadstring = loadstring or load

function File:isWritableObject(object)
   local typename = type(object)
   local typeidx
   if type(object) ~= 'boolean' and not object then
      typeidx = TYPE_NIL
   elseif torch.typename(object) and torch.factory(torch.typename(object)) then
      typeidx = TYPE_TORCH
   elseif typename == 'table' then
      typeidx = TYPE_TABLE
   elseif typename == 'number' then
      typeidx = TYPE_NUMBER
   elseif typename == 'string' then
      typeidx = TYPE_STRING
   elseif typename == 'boolean' then
      typeidx = TYPE_BOOLEAN
   elseif typename == 'function' and pcall(string.dump, object) then
      typeidx = TYPE_RECUR_FUNCTION
   end
   return typeidx
end

function File:referenced(ref)
   -- we use an environment to keep a record of written objects
   if not torch.getenv(self).writeObjects then
      torch.setenv(self, {
            writeObjects={}, writeObjectsRef={},
            readObjects={},
            objectNameStack={},
            upvalueRefToId={}, upvalueIdToClosure={},
         })
   end
   local env = torch.getenv(self)
   env.force = not ref
   torch.setenv(self,env)
   return self
end

function File:isReferenced()
   -- if no environment, then no forcing setup yet
   if not torch.getenv(self).writeObjects then
      return true
   end
   local env = torch.getenv(self)
   return not env.force
end

local function getmetamethod(obj, name)
   local func
   local status

   -- check getmetatable(obj).__name or
   -- check getmetatable(obj).name
   status, func = pcall(
      function()
         -- note that sometimes the metatable is hidden
         -- we get it for sure through the torch type system
         local mt = torch.getmetatable(torch.typename(obj))
         if mt then
            return mt['__' .. name] or mt[name]
         end
      end
   )
   if status and type(func) == 'function' then
      return func
   end
end

local UPVALUES_TOKEN = {} -- unique object
local function formatStack(objectNameStack)
   -- Format object name stack skipping UPVALUES_TOKEN and upvalue index
   local parts = {}
   for i, v in ipairs(objectNameStack) do
      if v ~= UPVALUES_TOKEN and objectNameStack[i-1] ~= UPVALUES_TOKEN then
         table.insert(parts, v)
      end
   end
   return table.concat(parts, '.')
end

function File:writeObject(object, debugname, hook)
   -- define a default hook function if not provided
   hook = hook or function(object) return object end
   -- we use an environment to keep a record of written objects
   if not torch.getenv(self).writeObjects then
      torch.setenv(self, {
            writeObjects={}, writeObjectsRef={},
            readObjects={},
            objectNameStack={},
            upvalueRefToId={}, upvalueIdToClosure={},
         })
   end
   -- That guy is used for references' book-keeping
   local sobject = object
   -- That guy is the object that is actually persisted
   -- hook(object) can be used to modify the object before writing it to the file.
   -- Useful for serializing objects under a config
   -- that we want to deserialize safely under another config.
   -- (e.g. Cuda to Float tensors, cudnn to nn, ...)
   object = hook(object)
   local force = torch.getenv(self).force

   -- if nil object, only write the type and return
   if type(object) ~= 'boolean' and not object then
      self:writeInt(TYPE_NIL)
      return
   end

   local objectNameStack = torch.getenv(self).objectNameStack
   table.insert(objectNameStack, debugname or '<?>')

   -- check the type we are dealing with
   local typeidx = self:isWritableObject(object)
   if not typeidx then
      error(string.format('Unwritable object <%s> at %s', type(object), formatStack(objectNameStack)))
   end
   self:writeInt(typeidx)

   if typeidx == TYPE_NUMBER then
      self:writeDouble(object)
   elseif typeidx == TYPE_BOOLEAN then
      self:writeBool(object)
   elseif typeidx == TYPE_STRING then
      local stringStorage = torch.CharStorage():string(object)
      self:writeInt(#stringStorage)
      self:writeChar(stringStorage)
   elseif typeidx == TYPE_TORCH or typeidx == TYPE_TABLE or  typeidx == TYPE_RECUR_FUNCTION then
      -- check it exists already (we look at the pointer!)
      local objects = torch.getenv(self).writeObjects
      local objectsRef = torch.getenv(self).writeObjectsRef
      local index = objects[torch.pointer(sobject)]

      if index and (not force) then
         -- if already exists, write only its index
         self:writeInt(index)
      else
         -- else write the object itself
         index = objects.nWriteObject or 0
         index = index + 1
         if not force then
            objects[torch.pointer(sobject)] = index
            objectsRef[object] = index -- we make sure the object is not going to disappear
         end
         self:writeInt(index)
         objects.nWriteObject = index
         if typeidx == TYPE_RECUR_FUNCTION then
            local upvalueRefToId = torch.getenv(self).upvalueRefToId
            -- Unique ID for each ref since lightuserdata are not serializable
            local nextId = 1
            for _ in pairs(upvalueRefToId) do nextId=nextId+1 end
            local upvalues = {}
            local counter = 0
            while true do
               counter = counter + 1
               local name,value = debug.getupvalue(object, counter)
               if not name then break end
               if name == '_ENV' then value = nil end
               local id=nil
               -- debug.upvalueid exists only for lua>=5.2 and luajit
               if debug.upvalueid then
                  local upvalueRef = debug.upvalueid(object, counter)
                  if not upvalueRefToId[upvalueRef] then
                     upvalueRefToId[upvalueRef] = nextId
                     nextId = nextId + 1
                  end
                  id = upvalueRefToId[upvalueRef]
               end
               table.insert(upvalues, {name=name, id=id, value=value})
            end
            local dumped = string.dump(object)
            local stringStorage = torch.CharStorage():string(dumped)
            self:writeInt(#stringStorage)
            self:writeChar(stringStorage)
            self:writeObject(upvalues, UPVALUES_TOKEN, hook)
         elseif typeidx == TYPE_TORCH then
            local version   = torch.CharStorage():string('V ' .. torch.version(object))
            local className = torch.CharStorage():string(torch.typename(object))
            self:writeInt(#version)
            self:writeChar(version)
            self:writeInt(#className)
            self:writeChar(className)
            local write = getmetamethod(object, 'write')
            if write then
               write(object, self)
            elseif type(object) == 'table' then
               local var = {}
               for k,v in pairs(object) do
                  if self:isWritableObject(v) then
                     var[k] = v
                  else
                     print(string.format('$ Warning: cannot write object field <%s> of <%s> %s', k, torch.typename(object), formatStack(objectNameStack)))
                  end
               end
               self:writeObject(var, torch.typename(object), hook)
            else
               error(string.format('<%s> is a non-serializable Torch object %s', torch.typename(object), formatStack(objectNameStack)))
            end
         else -- it is a table
            local size = 0; for k,v in pairs(object) do size = size + 1 end
            self:writeInt(size)
            for k,v in pairs(object) do
               self:writeObject(k, nil, hook)
               local name = (type(k) == 'string' or type(k) == 'number') and tostring(k) or nil
               -- special case name for upvalues
               if objectNameStack[#objectNameStack-1] == UPVALUES_TOKEN and
                  name == 'value' and type(object.name) == 'string' then
                  name = object.name
               end
               self:writeObject(v, name, hook)
            end
         end
      end
   else
      error('Unwritable object')
   end
   table.remove(objectNameStack)
end

function File:readObject()
   -- we use an environment to keep a record of read objects
   if not torch.getenv(self).writeObjects then
      torch.setenv(self, {
            writeObjects={}, writeObjectsRef={},
            readObjects={},
            objectNameStack={},
            upvalueRefToId={}, upvalueIdToClosure={},
         })
   end

   local force = torch.getenv(self).force

   -- read the typeidx
   local typeidx = self:readInt()

   -- is it nil?
   if typeidx == TYPE_NIL then
      return nil
   end

   if typeidx == TYPE_NUMBER then
      return self:readDouble()
   elseif typeidx == TYPE_BOOLEAN then
      return self:readBool()
   elseif typeidx == TYPE_STRING then
      local size = self:readInt()
      return self:readChar(size):string()
   elseif typeidx == TYPE_FUNCTION then
       local size = self:readInt()
       local dumped = self:readChar(size):string()
       local func, err = loadstring(dumped)
       if not func then
          error(string.format('Failed to load function from bytecode: %s', err))
       end
       local upvalues = self:readObject()
       for index,upvalue in ipairs(upvalues) do
          debug.setupvalue(func, index, upvalue)
       end
       return func
   elseif typeidx == TYPE_TABLE or typeidx == TYPE_TORCH or typeidx == TYPE_RECUR_FUNCTION or typeidx == LEGACY_TYPE_RECUR_FUNCTION then
      -- read the index
      local index = self:readInt()

      -- check it is loaded already
      local objects = torch.getenv(self).readObjects
      if objects[index] and not force then
         return objects[index]
      end

      -- otherwise read it
      if typeidx == TYPE_RECUR_FUNCTION or typeidx == LEGACY_TYPE_RECUR_FUNCTION then
         local size = self:readInt()
         local dumped = self:readChar(size):string()
         local func, err = loadstring(dumped)
         if not func then
            error(string.format('Failed to load function from bytecode: %s', err))
         end
         if not force then
             objects[index] = func
         end
         local upvalueIdToClosure = torch.getenv(self).upvalueIdToClosure
         local upvalues = self:readObject()
         for index,upvalue in ipairs(upvalues) do
            if typeidx == LEGACY_TYPE_RECUR_FUNCTION then
               debug.setupvalue(func, index, upvalue)
            elseif upvalue.name == '_ENV' then
               debug.setupvalue(func, index, _ENV)
            else
               debug.setupvalue(func, index, upvalue.value)
               -- debug.upvaluejoin exists only for lua>=5.2 and luajit
               if debug.upvaluejoin and upvalue.id then
                  if upvalueIdToClosure[upvalue.id] then
                     -- This upvalue is linked to another one
                     local otherClosure = upvalueIdToClosure[upvalue.id]
                     debug.upvaluejoin(func, index, otherClosure.func, otherClosure.index)
                  else
                     -- Save this closure for next time
                     upvalueIdToClosure[upvalue.id] = {
                        func = func,
                        index = index,
                     }
                  end
               end
            end
         end
         return func
      elseif typeidx == TYPE_TORCH then
         local version, className, versionNumber
         version = self:readChar(self:readInt()):string()
         versionNumber = tonumber(string.match(version, '^V (.*)$'))
         if not versionNumber then
            className = version
            versionNumber = 0 -- file created before existence of versioning system
         else
            className = self:readChar(self:readInt()):string()
         end
         if not torch.factory(className) then
            error(string.format('unknown Torch class <%s>', tostring(className)))
         end
         local object = torch.factory(className)(self)
         if not force then
             objects[index] = object
         end
         local read = getmetamethod(object, 'read')
         if read then
            read(object, self, versionNumber)
         elseif type(object) == 'table' then
            local var = self:readObject()
            for k,v in pairs(var) do
               object[k] = v
            end
         else
            error(string.format('Cannot load object class <%s>', tostring(className)))
         end
         return object
      else -- it is a table
         local size = self:readInt()
         local object = {}
         if not force then
             objects[index] = object
         end
         for i = 1,size do
            local k = self:readObject()
            local v = self:readObject()
            object[k] = v
         end
         return object
      end
   else
      error('unknown object')
   end
end

-- simple helpers to save/load arbitrary objects/tables
function torch.save(filename, object, mode, referenced)
   assert(mode == nil or mode == 'binary' or mode == 'ascii', '"binary" or "ascii" (or nil) expected for mode')
   assert(referenced == nil or referenced == true or referenced == false, 'true or false (or nil) expected for referenced')
   mode = mode or 'binary'
   referenced = referenced == nil and true or referenced
   local file = torch.DiskFile(filename, 'w')
   file[mode](file)
   file:referenced(referenced)
   file:writeObject(object)
   file:close()
end

function torch.load(filename, mode, referenced)
   assert(mode == nil or mode == 'binary' or mode == 'ascii', '"binary" or "ascii" (or nil) expected for mode')
   assert(referenced == nil or referenced == true or referenced == false, 'true or false (or nil) expected for referenced')
   mode = mode or 'binary'
   referenced = referenced == nil and true or referenced
   local file = torch.DiskFile(filename, 'r')
   file[mode](file)
   file:referenced(referenced)
   local object = file:readObject()
   file:close()
   return object
end

-- simple helpers to serialize/deserialize arbitrary objects/tables
function torch.serialize(object, mode)
   local storage = torch.serializeToStorage(object, mode)
   return storage:string()
end

-- Serialize to a CharStorage, not a lua string. This avoids
function torch.serializeToStorage(object, mode)
   mode = mode or 'binary'
   local f = torch.MemoryFile()
   f = f[mode](f)
   f:writeObject(object)
   local storage = f:storage()
   f:close()
   return storage
end

function torch.deserializeFromStorage(storage, mode)
   mode = mode or 'binary'
   local tx = torch.CharTensor(storage)
   local xp = torch.CharStorage(tx:size(1)+1)
   local txp = torch.CharTensor(xp)
   txp:narrow(1,1,tx:size(1)):copy(tx)
   txp[tx:size(1)+1] = 0
   local f = torch.MemoryFile(xp)
   f = f[mode](f)
   local object = f:readObject()
   f:close()
   return object
end

function torch.deserialize(str, mode)
   local storage = torch.CharStorage():string(str)
   return torch.deserializeFromStorage(storage, mode)
end

-- public API (saveobj/loadobj are safe for global import)
torch.saveobj = torch.save
torch.loadobj = torch.load
