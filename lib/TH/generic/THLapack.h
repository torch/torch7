#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THLapack.h"
#else

/* AX=B */
TH_API void THLapack_(gesv)(LAPACK_INT n, LAPACK_INT nrhs, real *a, LAPACK_INT lda, LAPACK_INT *ipiv, real *b, LAPACK_INT ldb, LAPACK_INT* info);
/* Solve a triangular system of the form A * X = B  or A^T * X = B */
TH_API void THLapack_(trtrs)(char uplo, char trans, char diag, LAPACK_INT n, LAPACK_INT nrhs, real *a, LAPACK_INT lda, real *b, LAPACK_INT ldb, LAPACK_INT* info);
/* ||AX-B|| */
TH_API void THLapack_(gels)(char trans, LAPACK_INT m, LAPACK_INT n, LAPACK_INT nrhs, real *a, LAPACK_INT lda, real *b, LAPACK_INT ldb, real *work, LAPACK_INT lwork, LAPACK_INT *info);
/* Eigenvals */
TH_API void THLapack_(syev)(char jobz, char uplo, LAPACK_INT n, real *a, LAPACK_INT lda, real *w, real *work, LAPACK_INT lwork, LAPACK_INT *info);
/* Non-sym eigenvals */
TH_API void THLapack_(geev)(char jobvl, char jobvr, LAPACK_INT n, real *a, LAPACK_INT lda, real *wr, real *wi, real* vl, LAPACK_INT ldvl, real *vr, LAPACK_INT ldvr, real *work, LAPACK_INT lwork, LAPACK_INT *info);
/* svd */
TH_API void THLapack_(gesvd)(char jobu, char jobvt, LAPACK_INT m, LAPACK_INT n, real *a, LAPACK_INT lda, real *s, real *u, LAPACK_INT ldu, real *vt, LAPACK_INT ldvt, real *work, LAPACK_INT lwork, LAPACK_INT *info);
/* LU decomposition */
TH_API void THLapack_(getrf)(LAPACK_INT m, LAPACK_INT n, real *a, LAPACK_INT lda, LAPACK_INT *ipiv, LAPACK_INT *info);
TH_API void THLapack_(getrs)(char trans, LAPACK_INT n, LAPACK_INT nrhs, real *a, LAPACK_INT lda, LAPACK_INT *ipiv, real *b, LAPACK_INT ldb, LAPACK_INT *info);
/* Matrix Inverse */
TH_API void THLapack_(getri)(LAPACK_INT n, real *a, LAPACK_INT lda, LAPACK_INT *ipiv, real *work, LAPACK_INT lwork, LAPACK_INT* info);

/* Positive Definite matrices */
/* Cholesky factorization */
void THLapack_(potrf)(char uplo, LAPACK_INT n, real *a, LAPACK_INT lda, LAPACK_INT *info);
/* Matrix inverse based on Cholesky factorization */
void THLapack_(potri)(char uplo, LAPACK_INT n, real *a, LAPACK_INT lda, LAPACK_INT *info);
/* Solve A*X = B with a symmetric positive definite matrix A using the Cholesky factorization */
void THLapack_(potrs)(char uplo, LAPACK_INT n, LAPACK_INT nrhs, real *a, LAPACK_INT lda, real *b, LAPACK_INT ldb, LAPACK_INT *info);
/* Cholesky factorization with complete pivoting. */
void THLapack_(pstrf)(char uplo, LAPACK_INT n, real *a, LAPACK_INT lda, LAPACK_INT *piv, LAPACK_INT *rank, real tol, real *work, LAPACK_INT *info);

/* QR decomposition */
void THLapack_(geqrf)(LAPACK_INT m, LAPACK_INT n, real *a, LAPACK_INT lda, real *tau, real *work, LAPACK_INT lwork, LAPACK_INT *info);
/* Build Q from output of geqrf */
void THLapack_(orgqr)(LAPACK_INT m, LAPACK_INT n, LAPACK_INT k, real *a, LAPACK_INT lda, real *tau, real *work, LAPACK_INT lwork, LAPACK_INT *info);
/* Multiply Q with a matrix from output of geqrf */
void THLapack_(ormqr)(char side, char trans, LAPACK_INT m, LAPACK_INT n, LAPACK_INT k, real *a, LAPACK_INT lda, real *tau, real *c, LAPACK_INT ldc, real *work, LAPACK_INT lwork, LAPACK_INT *info);

#endif
