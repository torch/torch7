#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THTensorLapack.h"
#else

TH_API void THTensor_(gesv)(THTensor *rb_, THTensor *ra_, THTensor *b_, THTensor *a_);
TH_API void THTensor_(trtrs)(THTensor *rb_, THTensor *ra_, THTensor *b_, THTensor *a_, const char *uplo, const char *trans, const char *diag);
TH_API void THTensor_(gels)(THTensor *rb_, THTensor *ra_, THTensor *b_, THTensor *a_);
TH_API void THTensor_(syev)(THTensor *re_, THTensor *rv_, THTensor *a_, const char *jobz, const char *uplo);
TH_API void THTensor_(geev)(THTensor *re_, THTensor *rv_, THTensor *a_, const char *jobvr);
TH_API void THTensor_(gesvd)(THTensor *ru_, THTensor *rs_, THTensor *rv_, THTensor *a, const char *jobu);
TH_API void THTensor_(gesvd2)(THTensor *ru_, THTensor *rs_, THTensor *rv_, THTensor *ra_, THTensor *a, const char *jobu);
TH_API void THTensor_(getri)(THTensor *ra_, THTensor *a);
TH_API void THTensor_(potrf)(THTensor *ra_, THTensor *a, const char *uplo);
TH_API void THTensor_(potrs)(THTensor *rb_, THTensor *b_, THTensor *a_,  const char *uplo);
TH_API void THTensor_(potri)(THTensor *ra_, THTensor *a, const char *uplo);
TH_API void THTensor_(qr)(THTensor *rq_, THTensor *rr_, THTensor *a);
TH_API void THTensor_(geqrf)(THTensor *ra_, THTensor *rtau_, THTensor *a);
TH_API void THTensor_(orgqr)(THTensor *ra_, THTensor *a, THTensor *tau);
TH_API void THTensor_(ormqr)(THTensor *ra_, THTensor *a, THTensor *tau, THTensor *c, const char *side, const char *trans);
TH_API void THTensor_(pstrf)(THTensor *ra_, THIntTensor *rpiv_, THTensor*a, const char* uplo, real tol);

#endif
