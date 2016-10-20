#include <stdio.h>
#include <stddef.h>

#include <ThreadPool_config.h>
#include <TPI.h>
#include <tpi_vector.h>

#if defined( HAVE_MPI )
#include <mpi.h>
#endif

/*--------------------------------------------------------------------*/

struct tpi_work_vector {
        VECTOR_SCALAR alpha ;
        VECTOR_SCALAR beta ;
  const VECTOR_SCALAR * x ;
  const VECTOR_SCALAR * y ;
        VECTOR_SCALAR * w ; 
        int  n ;
};

void tpi_work_span( TPI_Work * const work , const int n ,
                    int * const iBeg , int * const iEnd )
{
  const int chunk = ( n + work->count - 1 ) / work->count ;
  const int i_end = chunk + ( *iBeg = chunk * work->rank );

  *iEnd = n < i_end ? n : i_end ;
}

/*--------------------------------------------------------------------*/

static void tpi_work_fill( TPI_Work * work )
{
  const struct tpi_work_vector * const h =
    (struct tpi_work_vector *) work->info ;

  const VECTOR_SCALAR alpha = h->alpha ;
  VECTOR_SCALAR * const w = h->w ;

  int i , iEnd ;

  tpi_work_span( work , h->n , & i , & iEnd );

  for ( ; i < iEnd ; ++i ) { w[i] = alpha ; }
}

void tpi_fill( int n , VECTOR_SCALAR alpha , VECTOR_SCALAR * x )
{
  struct tpi_work_vector tmp = { 0.0 , 0.0 , NULL , NULL , NULL , 0 };
  tmp.alpha = alpha ;
  tmp.w = x ;
  tmp.n = n ;
  TPI_Run_threads( tpi_work_fill , & tmp , 0 );
}

/*--------------------------------------------------------------------*/

static void tpi_work_scale( TPI_Work * work )
{
  const struct tpi_work_vector * const h =
    (struct tpi_work_vector *) work->info ;

  const VECTOR_SCALAR beta = h->beta ;
  VECTOR_SCALAR * const w = h->w ;

  int i , iEnd ;

  tpi_work_span( work , h->n , & i , & iEnd );

  for ( ; i < iEnd ; ++i ) { w[i] *= beta ; }
}

void tpi_scale( int n , const VECTOR_SCALAR alpha , VECTOR_SCALAR * x )
{
  struct tpi_work_vector tmp = { 0.0 , 0.0 , NULL , NULL , NULL , 0 };
  tmp.alpha = alpha ;
  tmp.w = x ;
  tmp.n = n ;
  TPI_Run_threads( tpi_work_scale , & tmp , 0 );
}

/*--------------------------------------------------------------------*/

static void tpi_work_copy( TPI_Work * work )
{
  const struct tpi_work_vector * const h =
    (struct tpi_work_vector *) work->info ;

  const VECTOR_SCALAR * const x = h->x ;
  VECTOR_SCALAR * const w = h->w ;

  int i , iEnd ;

  tpi_work_span( work , h->n , & i , & iEnd );

  for ( ; i < iEnd ; ++i ) { w[i] = x[i] ; }
}

void tpi_copy( int n , const VECTOR_SCALAR * x , VECTOR_SCALAR * y )
{
  struct tpi_work_vector tmp = { 0.0 , 0.0 , NULL , NULL , NULL , 0 };
  tmp.x = x ;
  tmp.w = y ;
  tmp.n = n ;
  TPI_Run_threads( tpi_work_copy , & tmp , 0 );
}

/*--------------------------------------------------------------------*/

static void tpi_work_axpby( TPI_Work * work )
{
  const struct tpi_work_vector * const h =
    (struct tpi_work_vector *) work->info ;

  const VECTOR_SCALAR alpha = h->alpha ;
  const VECTOR_SCALAR beta  = h->beta ;
  const VECTOR_SCALAR * const x = h->x ;
  VECTOR_SCALAR * const w = h->w ;

  int i , iEnd ;

  tpi_work_span( work , h->n , & i , & iEnd );

  for ( ; i < iEnd ; ++i ) { w[i] = alpha * x[i] + beta * w[i] ; }
}

void tpi_axpby( int n , VECTOR_SCALAR alpha , const VECTOR_SCALAR * x ,
                        VECTOR_SCALAR beta  ,       VECTOR_SCALAR * y )
{
  struct tpi_work_vector tmp = { 0.0 , 0.0 , NULL , NULL , NULL , 0 };
  tmp.alpha = alpha ;
  tmp.beta  = beta ;
  tmp.x = x ;
  tmp.w = y ;
  tmp.n = n ;

  TPI_Run_threads( tpi_work_axpby , & tmp , 0 );
}

/*--------------------------------------------------------------------*/

static void tpi_work_axpy( TPI_Work * work )
{
  const struct tpi_work_vector * const h =
    (struct tpi_work_vector *) work->info ;

  const VECTOR_SCALAR alpha = h->alpha ;
  const VECTOR_SCALAR * const x = h->x ;
  VECTOR_SCALAR * const w = h->w ;

  int i , iEnd ;

  tpi_work_span( work , h->n , & i , & iEnd );

  for ( ; i < iEnd ; ++i ) { w[i] += alpha * x[i] ; }
}

void tpi_axpy( int n , VECTOR_SCALAR alpha , const VECTOR_SCALAR * x ,
                                                   VECTOR_SCALAR * y )
{
  struct tpi_work_vector tmp = { 0.0 , 0.0 , NULL , NULL , NULL , 0 };
  tmp.alpha = alpha ;
  tmp.x = x ;
  tmp.w = y ;
  tmp.n = n ;

  TPI_Run_threads( tpi_work_axpy , & tmp , 0 );
}

/*--------------------------------------------------------------------*/

static void tpi_work_xpby( TPI_Work * work )
{
  const struct tpi_work_vector * const h =
    (struct tpi_work_vector *) work->info ;

  const VECTOR_SCALAR beta  = h->beta ;
  const VECTOR_SCALAR * const x = h->x ;
  VECTOR_SCALAR * const w = h->w ;

  int i , iEnd ;

  tpi_work_span( work , h->n , & i , & iEnd );

  for ( ; i < iEnd ; ++i ) { w[i] = x[i] + beta * w[i] ; }
}

void tpi_xpby( int n , const VECTOR_SCALAR * x , VECTOR_SCALAR beta  ,
                                                 VECTOR_SCALAR * y )
{
  struct tpi_work_vector tmp = { 0.0 , 0.0 , NULL , NULL , NULL , 0 };
  tmp.beta  = beta ;
  tmp.x = x ;
  tmp.w = y ;
  tmp.n = n ;

  TPI_Run_threads( tpi_work_xpby , & tmp , 0 );
}

/*--------------------------------------------------------------------*/

static void tpi_work_dot_partial( TPI_Work * work )
{
  const struct tpi_work_vector * const h =
    (struct tpi_work_vector *) work->info ;

  const VECTOR_SCALAR * const x = h->x ;
  const VECTOR_SCALAR * const y = h->y ;
  double * const s = (double *) work->reduce ;
  double tmp = *s ;
  int i , iEnd ;

  tpi_work_span( work , h->n , & i , & iEnd );

  for ( ; i < iEnd ; ++i ) { tmp += x[i] * y[i] ; }

  *s = tmp ;
}

static void tpi_work_dot_partial_self( TPI_Work * work )
{
  const struct tpi_work_vector * const h =
    (struct tpi_work_vector *) work->info ;

  const VECTOR_SCALAR * const x = h->x ;
  double * const s = (double *) work->reduce ;
  double tmp = *s ;

  int i , iEnd ;

  tpi_work_span( work , h->n , & i , & iEnd );

  for ( ; i < iEnd ; ++i ) { const VECTOR_SCALAR d = x[i] ; tmp += d * d ; }

  *s = tmp ;
}

static void tpi_work_dot_join( TPI_Work * work , const void * src  )
{
  *((double *) ( work->reduce) ) += *((const double *) src);
}

static void tpi_work_dot_init( TPI_Work * work )
{
  *((double *) ( work->reduce) ) = 0 ;
}

double tpi_dot( int n , const VECTOR_SCALAR * x , const VECTOR_SCALAR * y )
{
  struct tpi_work_vector tmp = { 0.0 , 0.0 , NULL , NULL , NULL , 0 };
  double result = 0.0 ;
  tmp.x = x ;
  tmp.y = y ;
  tmp.n = n ;
  if ( x != y ) {
    TPI_Run_threads_reduce( tpi_work_dot_partial , & tmp ,
                            tpi_work_dot_join , tpi_work_dot_init ,
                            sizeof(result) , & result );
  }
  else {
    TPI_Run_threads_reduce( tpi_work_dot_partial_self , & tmp ,
                            tpi_work_dot_join , tpi_work_dot_init ,
                            sizeof(result) , & result );
  }
#if defined HAVE_MPI
  {
    double tmp = result ;
    MPI_Allreduce( & tmp , & result , 1 , MPI_DOUBLE , MPI_SUM , MPI_COMM_WORLD );
  }
#endif
  return result ;
}

/*--------------------------------------------------------------------*/

