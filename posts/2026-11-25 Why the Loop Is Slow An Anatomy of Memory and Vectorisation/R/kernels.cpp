// Compiled kernels. Two of the experiments (pointer chasing and traversal
// order) only make sense without the R interpreter in the way.

#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
double sum_cpp(NumericVector x, NumericVector y) {
  double s = 0;
  for (R_xlen_t i = 0; i < x.size(); ++i) s += x[i] * y[i];
  return s;
}

// [[Rcpp::export]]
double sum1(NumericVector x) {
  double s = 0;
  for (R_xlen_t i = 0; i < x.size(); ++i) s += x[i];
  return s;
}

// Same loop, but walking a raw pointer instead of the vector accessor.
// [[Rcpp::export]]
double sum1_ptr(NumericVector x) {
  const double* p = x.begin();
  double s = 0;
  R_xlen_t n = x.size();
  for (R_xlen_t i = 0; i < n; ++i) s += p[i];
  return s;
}

// One output vector, no temporaries: the memory-lean way to combine three.
// [[Rcpp::export]]
NumericVector hypot3(NumericVector x, NumericVector y, NumericVector z) {
  R_xlen_t n = x.size();
  NumericVector out(n);
  for (R_xlen_t i = 0; i < n; ++i)
    out[i] = x[i] * x[i] + y[i] * y[i] + z[i] * z[i];
  return out;
}

// Sum elements at random positions. The index vector is drawn once so that
// the only thing changing between runs is where the data sits in memory.
// R indices are 1-based, C++ ones are not, hence the -1.
// [[Rcpp::export]]
double sum_at(NumericVector x, IntegerVector idx) {
  double s = 0;
  for (R_xlen_t k = 0; k < idx.size(); ++k) s += x[idx[k] - 1];
  return s;
}

// Traverse a matrix in memory order (column by column, as R stores it).
// [[Rcpp::export]]
double sum_col_major(NumericMatrix m) {
  double s = 0;
  int nr = m.nrow(), nc = m.ncol();
  for (int j = 0; j < nc; ++j)
    for (int i = 0; i < nr; ++i) s += m(i, j);
  return s;
}

// Traverse the same matrix row by row, i.e. against the memory layout.
// [[Rcpp::export]]
double sum_row_major(NumericMatrix m) {
  double s = 0;
  int nr = m.nrow(), nc = m.ncol();
  for (int i = 0; i < nr; ++i)
    for (int j = 0; j < nc; ++j) s += m(i, j);
  return s;
}
