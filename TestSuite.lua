function torch.TestSuite()
   local obj = {
      __tests = {},
      __isTestSuite = true
   }

   local metatable = {}

   function metatable:__index(key)
      return self.__tests[key]
   end

   function metatable:__newindex(key, value)
      if self.__tests[key] ~= nil then
         error("Test " .. tostring(key) .. " is already defined.")
      end
      if type(value) ~= "function" then
         if type(value) == "table" then
            error("Nested tables of tests are not supported")
         else
            error("Only functions are supported as members of a TestSuite")
         end
      end
      self.__tests[key] = value
   end

   setmetatable(obj, metatable)

   return obj
end
