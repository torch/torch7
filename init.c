#include "general.h"
#include "utils.h"

extern void torch_utils_init(lua_State *L);
extern void torch_random_init(lua_State *L);
extern void torch_File_init(lua_State *L);
extern void torch_DiskFile_init(lua_State *L);
extern void torch_MemoryFile_init(lua_State *L);
extern void torch_PipeFile_init(lua_State *L);
extern void torch_Timer_init(lua_State *L);

extern void torch_ByteStorage_init(lua_State *L);
extern void torch_CharStorage_init(lua_State *L);
extern void torch_ShortStorage_init(lua_State *L);
extern void torch_IntStorage_init(lua_State *L);
extern void torch_LongStorage_init(lua_State *L);
extern void torch_FloatStorage_init(lua_State *L);
extern void torch_DoubleStorage_init(lua_State *L);

extern void torch_ByteTensor_init(lua_State *L);
extern void torch_CharTensor_init(lua_State *L);
extern void torch_ShortTensor_init(lua_State *L);
extern void torch_IntTensor_init(lua_State *L);
extern void torch_LongTensor_init(lua_State *L);
extern void torch_FloatTensor_init(lua_State *L);
extern void torch_DoubleTensor_init(lua_State *L);

extern void torch_ByteTensorOperator_init(lua_State *L);
extern void torch_CharTensorOperator_init(lua_State *L);
extern void torch_ShortTensorOperator_init(lua_State *L);
extern void torch_IntTensorOperator_init(lua_State *L);
extern void torch_LongTensorOperator_init(lua_State *L);
extern void torch_FloatTensorOperator_init(lua_State *L);
extern void torch_DoubleTensorOperator_init(lua_State *L);

extern void torch_TensorMath_init(lua_State *L);

LUAT_API int luaopen_libtorch(lua_State *L);

int luaopen_libtorch(lua_State *L)
{

  lua_newtable(L);
  lua_pushvalue(L, -1);
  lua_setglobal(L, "torch");

  torch_utils_init(L);

  torch_File_init(L);

  torch_ByteStorage_init(L);
  torch_CharStorage_init(L);
  torch_ShortStorage_init(L);
  torch_IntStorage_init(L);
  torch_LongStorage_init(L);
  torch_FloatStorage_init(L);
  torch_DoubleStorage_init(L);

  torch_ByteTensor_init(L);
  torch_CharTensor_init(L);
  torch_ShortTensor_init(L);
  torch_IntTensor_init(L);
  torch_LongTensor_init(L);
  torch_FloatTensor_init(L);
  torch_DoubleTensor_init(L);

  torch_ByteTensorOperator_init(L);
  torch_CharTensorOperator_init(L);
  torch_ShortTensorOperator_init(L);
  torch_IntTensorOperator_init(L);
  torch_LongTensorOperator_init(L);
  torch_FloatTensorOperator_init(L);
  torch_DoubleTensorOperator_init(L);

  torch_Timer_init(L);
  torch_DiskFile_init(L);
  torch_PipeFile_init(L);
  torch_MemoryFile_init(L);

  torch_TensorMath_init(L);

  torch_random_init(L);

  // Create 'torch.Allocator' type.
  luaT_newmetatable(L, "torch.Allocator", NULL, NULL, NULL, NULL);

  return 1;
}
