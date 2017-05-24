#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/THLapack.c"
#else

TH_EXTERNC void dgesv_(LAPACK_INT *n, LAPACK_INT *nrhs, double *a, LAPACK_INT *lda, LAPACK_INT *ipiv, double *b, LAPACK_INT *ldb, LAPACK_INT *info);
TH_EXTERNC void sgesv_(LAPACK_INT *n, LAPACK_INT *nrhs, float *a, LAPACK_INT *lda, LAPACK_INT *ipiv, float *b, LAPACK_INT *ldb, LAPACK_INT *info);
TH_EXTERNC void dtrtrs_(char *uplo, char *trans, char *diag, LAPACK_INT *n, LAPACK_INT *nrhs, double *a, LAPACK_INT *lda, double *b, LAPACK_INT *ldb, LAPACK_INT *info);
TH_EXTERNC void strtrs_(char *uplo, char *trans, char *diag, LAPACK_INT *n, LAPACK_INT *nrhs, float *a, LAPACK_INT *lda, float *b, LAPACK_INT *ldb, LAPACK_INT *info);
TH_EXTERNC void dgels_(char *trans, LAPACK_INT *m, LAPACK_INT *n, LAPACK_INT *nrhs, double *a, LAPACK_INT *lda, double *b, LAPACK_INT *ldb, double *work, LAPACK_INT *lwork, LAPACK_INT *info);
TH_EXTERNC void sgels_(char *trans, LAPACK_INT *m, LAPACK_INT *n, LAPACK_INT *nrhs, float *a, LAPACK_INT *lda, float *b, LAPACK_INT *ldb, float *work, LAPACK_INT *lwork, LAPACK_INT *info);
TH_EXTERNC void dsyev_(char *jobz, char *uplo, LAPACK_INT *n, double *a, LAPACK_INT *lda, double *w, double *work, LAPACK_INT *lwork, LAPACK_INT *info);
TH_EXTERNC void ssyev_(char *jobz, char *uplo, LAPACK_INT *n, float *a, LAPACK_INT *lda, float *w, float *work, LAPACK_INT *lwork, LAPACK_INT *info);
TH_EXTERNC void dgeev_(char *jobvl, char *jobvr, LAPACK_INT *n, double *a, LAPACK_INT *lda, double *wr, double *wi, double* vl, LAPACK_INT *ldvl, double *vr, LAPACK_INT *ldvr, double *work, LAPACK_INT *lwork, LAPACK_INT *info);
TH_EXTERNC void sgeev_(char *jobvl, char *jobvr, LAPACK_INT *n, float *a, LAPACK_INT *lda, float *wr, float *wi, float* vl, LAPACK_INT *ldvl, float *vr, LAPACK_INT *ldvr, float *work, LAPACK_INT *lwork, LAPACK_INT *info);
TH_EXTERNC void dgesvd_(char *jobu, char *jobvt, LAPACK_INT *m, LAPACK_INT *n, double *a, LAPACK_INT *lda, double *s, double *u, LAPACK_INT *ldu, double *vt, LAPACK_INT *ldvt, double *work, LAPACK_INT *lwork, LAPACK_INT *info);
TH_EXTERNC void sgesvd_(char *jobu, char *jobvt, LAPACK_INT *m, LAPACK_INT *n, float *a, LAPACK_INT *lda, float *s, float *u, LAPACK_INT *ldu, float *vt, LAPACK_INT *ldvt, float *work, LAPACK_INT *lwork, LAPACK_INT *info);
TH_EXTERNC void dgetrf_(LAPACK_INT *m, LAPACK_INT *n, double *a, LAPACK_INT *lda, LAPACK_INT *ipiv, LAPACK_INT *info);
TH_EXTERNC void sgetrf_(LAPACK_INT *m, LAPACK_INT *n, float *a, LAPACK_INT *lda, LAPACK_INT *ipiv, LAPACK_INT *info);
TH_EXTERNC void dgetrs_(char *trans, LAPACK_INT *n, LAPACK_INT *nrhs, double *a, LAPACK_INT *lda, LAPACK_INT *ipiv, double *b, LAPACK_INT *ldb, LAPACK_INT *info);
TH_EXTERNC void sgetrs_(char *trans, LAPACK_INT *n, LAPACK_INT *nrhs, float *a, LAPACK_INT *lda, LAPACK_INT *ipiv, float *b, LAPACK_INT *ldb, LAPACK_INT *info);
TH_EXTERNC void dgetri_(LAPACK_INT *n, double *a, LAPACK_INT *lda, LAPACK_INT *ipiv, double *work, LAPACK_INT *lwork, LAPACK_INT *info);
TH_EXTERNC void sgetri_(LAPACK_INT *n, float *a, LAPACK_INT *lda, LAPACK_INT *ipiv, float *work, LAPACK_INT *lwork, LAPACK_INT *info);
TH_EXTERNC void dpotrf_(char *uplo, LAPACK_INT *n, double *a, LAPACK_INT *lda, LAPACK_INT *info);
TH_EXTERNC void spotrf_(char *uplo, LAPACK_INT *n, float *a, LAPACK_INT *lda, LAPACK_INT *info);
TH_EXTERNC void dpotri_(char *uplo, LAPACK_INT *n, double *a, LAPACK_INT *lda, LAPACK_INT *info);
TH_EXTERNC void spotri_(char *uplo, LAPACK_INT *n, float *a, LAPACK_INT *lda, LAPACK_INT *info);
TH_EXTERNC void dpotrs_(char *uplo, LAPACK_INT *n, LAPACK_INT *nrhs, double *a, LAPACK_INT *lda, double *b, LAPACK_INT *ldb, LAPACK_INT *info);
TH_EXTERNC void spotrs_(char *uplo, LAPACK_INT *n, LAPACK_INT *nrhs, float *a, LAPACK_INT *lda, float *b, LAPACK_INT *ldb, LAPACK_INT *info);
TH_EXTERNC void sgeqrf_(LAPACK_INT *m, LAPACK_INT *n, float *a, LAPACK_INT *lda, float *tau, float *work, LAPACK_INT *lwork, LAPACK_INT *info);
TH_EXTERNC void dgeqrf_(LAPACK_INT *m, LAPACK_INT *n, double *a, LAPACK_INT *lda, double *tau, double *work, LAPACK_INT *lwork, LAPACK_INT *info);
TH_EXTERNC void sorgqr_(LAPACK_INT *m, LAPACK_INT *n, LAPACK_INT *k, float *a, LAPACK_INT *lda, float *tau, float *work, LAPACK_INT *lwork, LAPACK_INT *info);
TH_EXTERNC void dorgqr_(LAPACK_INT *m, LAPACK_INT *n, LAPACK_INT *k, double *a, LAPACK_INT *lda, double *tau, double *work, LAPACK_INT *lwork, LAPACK_INT *info);
TH_EXTERNC void sormqr_(char *side, char *trans, LAPACK_INT *m, LAPACK_INT *n, LAPACK_INT *k, float *a, LAPACK_INT *lda, float *tau, float *c, LAPACK_INT *ldc, float *work, LAPACK_INT *lwork, LAPACK_INT *info);
TH_EXTERNC void dormqr_(char *side, char *trans, LAPACK_INT *m, LAPACK_INT *n, LAPACK_INT *k, double *a, LAPACK_INT *lda, double *tau, double *c, LAPACK_INT *ldc, double *work, LAPACK_INT *lwork, LAPACK_INT *info);
TH_EXTERNC void spstrf_(char *uplo, LAPACK_INT *n, float *a, LAPACK_INT *lda, LAPACK_INT *piv, LAPACK_INT *rank, float *tol, float *work, LAPACK_INT *info);
TH_EXTERNC void dpstrf_(char *uplo, LAPACK_INT *n, double *a, LAPACK_INT *lda, LAPACK_INT *piv, LAPACK_INT *rank, double *tol, double *work, LAPACK_INT *info);


/* Compute the solution to a real system of linear equations  A * X = B */
void THLapack_(gesv)(LAPACK_INT n, LAPACK_INT nrhs, real *a, LAPACK_INT lda, LAPACK_INT *ipiv, real *b, LAPACK_INT ldb, LAPACK_INT* info)
{
#ifdef USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
#else
  sgesv_(&n, &nrhs, a, &lda, ipiv, b, &ldb, info);
#endif
#else
  THError("gesv : Lapack library not found in compile time\n");
#endif
  return;
}

/* Solve a triangular system of the form A * X = B  or A^T * X = B */
void THLapack_(trtrs)(char uplo, char trans, char diag, LAPACK_INT n, LAPACK_INT nrhs, real *a, LAPACK_INT lda, real *b, LAPACK_INT ldb, LAPACK_INT* info)
{
#ifdef USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dtrtrs_(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, info);
#else
  strtrs_(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, info);
#endif
#else
  THError("trtrs : Lapack library not found in compile time\n");
#endif
  return;
}

/* Solve overdetermined or underdetermined real linear systems involving an
M-by-N matrix A, or its transpose, using a QR or LQ factorization of A */
void THLapack_(gels)(char trans, LAPACK_INT m, LAPACK_INT n, LAPACK_INT nrhs, real *a, LAPACK_INT lda, real *b, LAPACK_INT ldb, real *work, LAPACK_INT lwork, LAPACK_INT *info)
{
#ifdef USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dgels_(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, info);
#else
  sgels_(&trans, &m, &n, &nrhs, a, &lda, b, &ldb, work, &lwork, info);
#endif
#else
  THError("gels : Lapack library not found in compile time\n");
#endif
}

/* Compute all eigenvalues and, optionally, eigenvectors of a real symmetric
matrix A */
void THLapack_(syev)(char jobz, char uplo, LAPACK_INT n, real *a, LAPACK_INT lda, real *w, real *work, LAPACK_INT lwork, LAPACK_INT *info)
{
#ifdef USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dsyev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, info);
#else
  ssyev_(&jobz, &uplo, &n, a, &lda, w, work, &lwork, info);
#endif
#else
  THError("syev : Lapack library not found in compile time\n");
#endif
}

/* Compute for an N-by-N real nonsymmetric matrix A, the eigenvalues and,
optionally, the left and/or right eigenvectors */
void THLapack_(geev)(char jobvl, char jobvr, LAPACK_INT n, real *a, LAPACK_INT lda, real *wr, real *wi, real* vl, LAPACK_INT ldvl, real *vr, LAPACK_INT ldvr, real *work, LAPACK_INT lwork, LAPACK_INT *info)
{
#ifdef USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, info);
#else
  sgeev_(&jobvl, &jobvr, &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr, work, &lwork, info);
#endif
#else
  THError("geev : Lapack library not found in compile time\n");
#endif
}

/* Compute the singular value decomposition (SVD) of a real M-by-N matrix A,
optionally computing the left and/or right singular vectors */
void THLapack_(gesvd)(char jobu, char jobvt, LAPACK_INT m, LAPACK_INT n, real *a, LAPACK_INT lda, real *s, real *u, LAPACK_INT ldu, real *vt, LAPACK_INT ldvt, real *work, LAPACK_INT lwork, LAPACK_INT *info)
{
#ifdef USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dgesvd_( &jobu,  &jobvt,  &m,  &n,  a,  &lda,  s,  u,  &ldu,  vt,  &ldvt,  work,  &lwork,  info);
#else
  sgesvd_( &jobu,  &jobvt,  &m,  &n,  a,  &lda,  s,  u,  &ldu,  vt,  &ldvt,  work,  &lwork,  info);
#endif
#else
  THError("gesvd : Lapack library not found in compile time\n");
#endif
}

/* LU decomposition */
void THLapack_(getrf)(LAPACK_INT m, LAPACK_INT n, real *a, LAPACK_INT lda, LAPACK_INT *ipiv, LAPACK_INT *info)
{
#ifdef  USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dgetrf_(&m, &n, a, &lda, ipiv, info);
#else
  sgetrf_(&m, &n, a, &lda, ipiv, info);
#endif
#else
  THError("getrf : Lapack library not found in compile time\n");
#endif
}

void THLapack_(getrs)(char trans, LAPACK_INT n, LAPACK_INT nrhs, real *a, LAPACK_INT lda, LAPACK_INT *ipiv, real *b, LAPACK_INT ldb, LAPACK_INT *info)
{
#ifdef  USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
#else
  sgetrs_(&trans, &n, &nrhs, a, &lda, ipiv, b, &ldb, info);
#endif
#else
  THError("getrs : Lapack library not found in compile time\n");
#endif
}

/* Matrix Inverse */
void THLapack_(getri)(LAPACK_INT n, real *a, LAPACK_INT lda, LAPACK_INT *ipiv, real *work, LAPACK_INT lwork, LAPACK_INT* info)
{
#ifdef  USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dgetri_(&n, a, &lda, ipiv, work, &lwork, info);
#else
  sgetri_(&n, a, &lda, ipiv, work, &lwork, info);
#endif
#else
  THError("getri : Lapack library not found in compile time\n");
#endif
}

/* Cholesky factorization */
void THLapack_(potrf)(char uplo, LAPACK_INT n, real *a, LAPACK_INT lda, LAPACK_INT *info)
{
#ifdef  USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dpotrf_(&uplo, &n, a, &lda, info);
#else
  spotrf_(&uplo, &n, a, &lda, info);
#endif
#else
  THError("potrf : Lapack library not found in compile time\n");
#endif
}

/* Solve A*X = B with a symmetric positive definite matrix A using the Cholesky factorization */
void THLapack_(potrs)(char uplo, LAPACK_INT n, LAPACK_INT nrhs, real *a, LAPACK_INT lda, real *b, LAPACK_INT ldb, LAPACK_INT *info)
{
#ifdef  USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dpotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, info);
#else
  spotrs_(&uplo, &n, &nrhs, a, &lda, b, &ldb, info);
#endif
#else
  THError("potrs: Lapack library not found in compile time\n");
#endif
}

/* Cholesky factorization based Matrix Inverse */
void THLapack_(potri)(char uplo, LAPACK_INT n, real *a, LAPACK_INT lda, LAPACK_INT *info)
{
#ifdef  USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dpotri_(&uplo, &n, a, &lda, info);
#else
  spotri_(&uplo, &n, a, &lda, info);
#endif
#else
  THError("potri: Lapack library not found in compile time\n");
#endif
}

/* Cholesky factorization with complete pivoting */
void THLapack_(pstrf)(char uplo, LAPACK_INT n, real *a, LAPACK_INT lda, LAPACK_INT *piv, LAPACK_INT *rank, real tol, real *work, LAPACK_INT *info)
{
#ifdef  USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dpstrf_(&uplo, &n, a, &lda, piv, rank, &tol, work, info);
#else
  spstrf_(&uplo, &n, a, &lda, piv, rank, &tol, work, info);
#endif
#else
  THError("pstrf: Lapack library not found at compile time\n");
#endif
}

/* QR decomposition */
void THLapack_(geqrf)(LAPACK_INT m, LAPACK_INT n, real *a, LAPACK_INT lda, real *tau, real *work, LAPACK_INT lwork, LAPACK_INT *info)
{
#ifdef  USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dgeqrf_(&m, &n, a, &lda, tau, work, &lwork, info);
#else
  sgeqrf_(&m, &n, a, &lda, tau, work, &lwork, info);
#endif
#else
  THError("geqrf: Lapack library not found in compile time\n");
#endif
}

/* Build Q from output of geqrf */
void THLapack_(orgqr)(LAPACK_INT m, LAPACK_INT n, LAPACK_INT k, real *a, LAPACK_INT lda, real *tau, real *work, LAPACK_INT lwork, LAPACK_INT *info)
{
#ifdef  USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, info);
#else
  sorgqr_(&m, &n, &k, a, &lda, tau, work, &lwork, info);
#endif
#else
  THError("orgqr: Lapack library not found in compile time\n");
#endif
}

/* Multiply Q with a matrix using the output of geqrf */
void THLapack_(ormqr)(char side, char trans, LAPACK_INT m, LAPACK_INT n, LAPACK_INT k, real *a, LAPACK_INT lda, real *tau, real *c, LAPACK_INT ldc, real *work, LAPACK_INT lwork, LAPACK_INT *info)
{
#ifdef  USE_LAPACK
#if defined(TH_REAL_IS_DOUBLE)
  dormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, info);
#else
  sormqr_(&side, &trans, &m, &n, &k, a, &lda, tau, c, &ldc, work, &lwork, info);
#endif
#else
  THError("ormqr: Lapack library not found in compile time\n");
#endif
}


#endif
