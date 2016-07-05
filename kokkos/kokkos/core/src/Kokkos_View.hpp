/*
//@HEADER
// ************************************************************************
//
//   Kokkos: Manycore Performance-Portable Multidimensional Arrays
//              Copyright (2012) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_VIEW_HPP
#define KOKKOS_VIEW_HPP

#include <string>
#include <Kokkos_Macros.hpp>
#include <Kokkos_HostSpace.hpp>
#include <Kokkos_MemoryTraits.hpp>

#include <impl/Kokkos_StaticAssert.hpp>
#include <impl/Kokkos_Traits.hpp>
#include <impl/Kokkos_Shape.hpp>
#include <impl/Kokkos_AnalyzeShape.hpp>
#include <impl/Kokkos_ViewSupport.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

/** \brief  View specialization mapping of view traits to a specialization tag */
template< typename ScalarType , class ValueType ,
          class ArrayLayout , class uRank , class uRankDynamic ,
          class MemorySpace , class MemoryTraits >
struct ViewSpecialize ;

template< class DstViewSpecialize , class SrcViewSpecialize = void , class Enable = void >
struct ViewAssignment ;

} /* namespace Impl */
} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

/** \class ViewTraits
 *  \brief Traits class for accessing attributes of a View.
 *
 * This is an implementation detail of View.  It is only of interest
 * to developers implementing a new specialization of View.
 *
 * Template argument permutations:
 *   - View< DataType , Device , void         , void >
 *   - View< DataType , Device , MemoryTraits , void >
 *   - View< DataType , Device , void         , MemoryTraits >
 *   - View< DataType , ArrayLayout , Device  , void >
 *   - View< DataType , ArrayLayout , Device  , MemoryTraits >
 */
template< class DataType ,
          class Arg1 ,
          class Arg2 ,
          class Arg3 >
class ViewTraits {
private:

  // Arg1 is either Device or Layout, both of which must have 'typedef ... array_layout'.
  // If Arg1 is not Layout then Arg1 must be Device
  enum { Arg1IsDevice = ! Impl::is_same< Arg1 , typename Arg1::array_layout >::value };
  enum { Arg2IsDevice = ! Arg1IsDevice };

  // If Arg1 is device and Arg2 is not void then Arg2 is MemoryTraits.
  // If Arg1 is device and Arg2 is void and Arg3 is not void then Arg3 is MemoryTraits.
  // If Arg2 is device and Arg3 is not void then Arg3 is MemoryTraits.
  enum { Arg2IsVoid = Impl::is_same< Arg2 , void >::value };
  enum { Arg3IsVoid = Impl::is_same< Arg3 , void >::value };
  enum { Arg2IsMemory = ! Arg2IsVoid && Arg1IsDevice && Arg3IsVoid };
  enum { Arg3IsMemory = ! Arg3IsVoid && ( ( Arg1IsDevice && Arg2IsVoid ) || Arg2IsDevice ) };


  typedef typename Arg1::array_layout  ArrayLayout ;
  typedef typename Impl::if_c< Arg1IsDevice , Arg1 , Arg2 >::type::device_type  DeviceType ;

  typedef typename Impl::if_c< Arg2IsMemory , Arg2 ,
          typename Impl::if_c< Arg3IsMemory , Arg3 , MemoryManaged
          >::type >::type::memory_traits  MemoryTraits ;

  typedef Impl::AnalyzeShape<DataType> analysis ;

public:

  //------------------------------------
  // Data type traits:

  typedef DataType                            data_type ;
  typedef typename analysis::const_type       const_data_type ;
  typedef typename analysis::non_const_type   non_const_data_type ;

  //------------------------------------
  // Scalar type traits:

  typedef typename analysis::scalar_type            scalar_type ;
  typedef typename analysis::const_scalar_type      const_scalar_type ;
  typedef typename analysis::non_const_scalar_type  non_const_scalar_type ;

  //------------------------------------
  // Value type traits:

  typedef typename analysis::value_type            value_type ;
  typedef typename analysis::const_value_type      const_value_type ;
  typedef typename analysis::non_const_value_type  non_const_value_type ;

  //------------------------------------
  // Layout and shape traits:

  typedef typename Impl::StaticAssertSame< ArrayLayout , typename ArrayLayout ::array_layout >::type  array_layout ;

  typedef typename analysis::shape   shape_type ;

  enum { rank         = shape_type::rank };
  enum { rank_dynamic = shape_type::rank_dynamic };

  //------------------------------------
  // Device and memory space traits:

  typedef typename Impl::StaticAssertSame< DeviceType   , typename DeviceType  ::device_type   >::type  device_type ;
  typedef typename Impl::StaticAssertSame< MemoryTraits , typename MemoryTraits::memory_traits >::type  memory_traits ;

  typedef typename device_type::memory_space  memory_space ;
  typedef typename device_type::size_type     size_type ;

  enum { is_hostspace = Impl::is_same< memory_space , HostSpace >::value };
  enum { is_managed   = memory_traits::Unmanaged == 0 };

  //------------------------------------
  // Specialization:
  typedef typename
    Impl::ViewSpecialize< scalar_type ,
                          value_type ,
                          array_layout ,
                          Impl::unsigned_<rank> ,
                          Impl::unsigned_<rank_dynamic> ,
                          memory_space ,
                          memory_traits
                        >::type specialize ;
};

} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

/** \brief  Default view specialization has ScalarType == ValueType
 *          and LayoutLeft or LayoutRight.
 */
struct LayoutDefault ;

template< typename ScalarType , class Rank , class RankDynamic , class MemorySpace , class MemoryTraits >
struct ViewSpecialize< ScalarType , ScalarType ,
                       LayoutLeft , Rank , RankDynamic ,
                       MemorySpace , MemoryTraits >
{ typedef LayoutDefault type ; };

template< typename ScalarType , class Rank , class RankDynamic , class MemorySpace , class MemoryTraits >
struct ViewSpecialize< ScalarType , ScalarType ,
                       LayoutRight , Rank , RankDynamic ,
                       MemorySpace , MemoryTraits >
{ typedef LayoutDefault type ; };

} /* namespace Impl */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

/** \brief Types for compile-time detection of View usage errors */
namespace ViewError {

struct allocation_constructor_requires_managed {};
struct user_pointer_constructor_requires_unmanaged {};
struct device_shmem_constructor_requires_unmanaged {};

struct scalar_operator_called_from_non_scalar_view {};

} /* namespace ViewError */

//----------------------------------------------------------------------------
/** \brief  Enable view parentheses operator for
 *          match of layout and integral arguments.
 *          If correct rank define type from traits,
 *          otherwise define type as an error message.
 */
template< class ReturnType , class Traits , class Layout , unsigned Rank ,
          typename iType0 = int , typename iType1 = int ,
          typename iType2 = int , typename iType3 = int ,
          typename iType4 = int , typename iType5 = int ,
          typename iType6 = int , typename iType7 = int ,
          class Enable = void >
struct ViewEnableArrayOper ;

template< class ReturnType , class Traits , class Layout , unsigned Rank ,
          typename iType0 , typename iType1 ,
          typename iType2 , typename iType3 ,
          typename iType4 , typename iType5 ,
          typename iType6 , typename iType7 >
struct ViewEnableArrayOper<
   ReturnType , Traits , Layout , Rank ,
   iType0 , iType1 , iType2 , iType3 ,
   iType4 , iType5 , iType6 , iType7 ,
   typename enable_if<
     iType0(0) == 0 && iType1(0) == 0 && iType2(0) == 0 && iType3(0) == 0 &&
     iType4(0) == 0 && iType5(0) == 0 && iType6(0) == 0 && iType7(0) == 0 &&
     is_same< typename Traits::array_layout , Layout >::value &&
     ( unsigned(Traits::rank) == Rank )
   >::type >
{
  typedef ReturnType type ;
};

} /* namespace Impl */
} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

struct AllocateWithoutInitializing {};

namespace {
const AllocateWithoutInitializing allocate_without_initializing = AllocateWithoutInitializing();
}

/** \class View
 *  \brief View to an array of data.
 *
 * A View represents an array of one or more dimensions.
 * For details, please refer to Kokkos' tutorial materials.
 *
 * \section Kokkos_View_TemplateParameters Template parameters
 *
 * This class has both required and optional template parameters.  The
 * \c DataType parameter must always be provided, and must always be
 * first. The parameters \c Arg1Type, \c Arg2Type, and \c Arg3Type are
 * placeholders for different template parameters.  The default value
 * of the fifth template parameter \c Specialize suffices for most use
 * cases.  When explaining the template parameters, we won't refer to
 * \c Arg1Type, \c Arg2Type, and \c Arg3Type; instead, we will refer
 * to the valid categories of template parameters, in whatever order
 * they may occur.
 *
 * Valid ways in which template arguments may be specified:
 *   - View< DataType , Device >
 *   - View< DataType , Device ,        MemoryTraits >
 *   - View< DataType , Device , void , MemoryTraits >
 *   - View< DataType , Layout , Device >
 *   - View< DataType , Layout , Device , MemoryTraits >
 *
 * \tparam DataType (required) This indicates both the type of each
 *   entry of the array, and the combination of compile-time and
 *   run-time array dimension(s).  For example, <tt>double*</tt>
 *   indicates a one-dimensional array of \c double with run-time
 *   dimension, and <tt>int*[3]</tt> a two-dimensional array of \c int
 *   with run-time first dimension and compile-time second dimension
 *   (of 3).  In general, the run-time dimensions (if any) must go
 *   first, followed by zero or more compile-time dimensions.  For
 *   more examples, please refer to the tutorial materials.
 *
 * \tparam Device (required) The execution model for parallel
 *   operations.  Examples include Threads, OpenMP, Cuda, and Serial.
 *
 * \tparam Layout (optional) The array's layout in memory.  For
 *   example, LayoutLeft indicates a column-major (Fortran style)
 *   layout, and LayoutRight a row-major (C style) layout.  If not
 *   specified, this defaults to the preferred layout for the
 *   <tt>Device</tt>.
 *
 * \tparam MemoryTraits (optional) Assertion of the user's intended
 *   access behavior.  For example, RandomRead indicates read-only
 *   access with limited spatial locality, and Unmanaged lets users
 *   wrap externally allocated memory in a View without automatic
 *   deallocation.
 *
 * \section Kokkos_View_MT \c MemoryTraits discussion
 *
 * \subsection Kokkos_View_MT_Interp \c MemoryTraits interpretation depends on \c Device
 *
 * Some \c MemoryTraits options may have different interpretations for
 * different \c Device types.  For example, with the Cuda device,
 * RandomRead tells Kokkos to fetch the data through the texture
 * cache, whereas the non-GPU devices have no such hardware construct.
 *
 * \subsection Kokkos_View_MT_PrefUse Preferred use of \c MemoryTraits
 *
 * Users should defer applying the optional \c MemoryTraits parameter
 * until the point at which they actually plan to rely on it in a
 * computational kernel.  This minimizes the number of template
 * parameters exposed in their code, which reduces the cost of
 * compilation.  Users may always assign a View without specified
 * <tt>MemoryTraits</tt> to a compatible View with that specification.
 * For example:
 * \code
 * // Pass in the simplest types of View possible.
 * void 
 * doSomething (View<double*, Cuda> out, 
 *              View<const double*, Cuda> in) 
 * {
 *   // Assign the "generic" View in to a RandomRead View in_rr.
 *   // Note that RandomRead View objects must have const data.
 *   View<const double*, Cuda, RandomRead> in_rr = in;
 *   // ... do something with in_rr and out ... 
 * }
 * \endcode
 */
template< class DataType ,
          class Arg1Type ,        /* ArrayLayout or DeviceType */
          class Arg2Type = void , /* DeviceType or MemoryTraits */
          class Arg3Type = void , /* MemoryTraits */
          class Specialize =
            typename ViewTraits<DataType,Arg1Type,Arg2Type,Arg3Type>::specialize >
class View ;

template< class DataType ,
          class Arg1Type ,
          class Arg2Type ,
          class Arg3Type >
class View< DataType , Arg1Type , Arg2Type , Arg3Type , Impl::LayoutDefault >
  : public ViewTraits< DataType , Arg1Type , Arg2Type, Arg3Type >
{
public:

  typedef ViewTraits< DataType , Arg1Type , Arg2Type, Arg3Type > traits ;

private:

  // Assignment of compatible views requirement:
  template< class , class , class , class , class > friend class View ;

  // Assignment of compatible subview requirement:
  template< class , class , class > friend struct Impl::ViewAssignment ;

  typedef Impl::LayoutStride< typename traits::shape_type ,
                              typename traits::array_layout > stride_type ;

  typename traits::scalar_type * m_ptr_on_device ;
  typename traits::shape_type    m_shape ;
  stride_type                    m_stride ;
  
public:

  typedef View< typename traits::const_data_type ,
                typename traits::array_layout ,
                typename traits::device_type ,
                typename traits::memory_traits > const_type ;

  typedef View< typename traits::non_const_data_type ,
                typename traits::array_layout ,
                typename traits::device_type::host_mirror_device_type ,
                void > HostMirror ;

  //------------------------------------
  // Shape

  enum { Rank = traits::rank };

  KOKKOS_INLINE_FUNCTION typename traits::shape_type shape() const { return m_shape ; }
  KOKKOS_INLINE_FUNCTION typename traits::size_type dimension_0() const { return m_shape.N0 ; }
  KOKKOS_INLINE_FUNCTION typename traits::size_type dimension_1() const { return m_shape.N1 ; }
  KOKKOS_INLINE_FUNCTION typename traits::size_type dimension_2() const { return m_shape.N2 ; }
  KOKKOS_INLINE_FUNCTION typename traits::size_type dimension_3() const { return m_shape.N3 ; }
  KOKKOS_INLINE_FUNCTION typename traits::size_type dimension_4() const { return m_shape.N4 ; }
  KOKKOS_INLINE_FUNCTION typename traits::size_type dimension_5() const { return m_shape.N5 ; }
  KOKKOS_INLINE_FUNCTION typename traits::size_type dimension_6() const { return m_shape.N6 ; }
  KOKKOS_INLINE_FUNCTION typename traits::size_type dimension_7() const { return m_shape.N7 ; }
  KOKKOS_INLINE_FUNCTION typename traits::size_type size() const
  {
    return   m_shape.N0
           * m_shape.N1
           * m_shape.N2
           * m_shape.N3
           * m_shape.N4
           * m_shape.N5
           * m_shape.N6
           * m_shape.N7
           ;
  }

  template< typename iType >
  KOKKOS_INLINE_FUNCTION
  typename traits::size_type dimension( const iType & i ) const
    { return Impl::dimension( m_shape , i ); }

  //------------------------------------

private:

  template< class ViewRHS >
  KOKKOS_INLINE_FUNCTION
  void assign_compatible_view( const ViewRHS & rhs ,
                               typename Impl::enable_if< Impl::ViewAssignable< View , ViewRHS >::value >::type * = 0 )
  {
    typedef typename traits::shape_type    shape_type ;
    typedef typename traits::memory_space  memory_space ;
    typedef typename traits::memory_traits memory_traits ;

    Impl::ViewTracking< traits >::decrement( m_ptr_on_device );

    shape_type::assign( m_shape,
                        rhs.m_shape.N0 , rhs.m_shape.N1 , rhs.m_shape.N2 , rhs.m_shape.N3 ,
                        rhs.m_shape.N4 , rhs.m_shape.N5 , rhs.m_shape.N6 , rhs.m_shape.N7 );

    stride_type::assign( m_stride , rhs.m_stride.value );

    m_ptr_on_device = rhs.m_ptr_on_device ;

    Impl::ViewTracking< traits >::increment( m_ptr_on_device );
  }

public:

  //------------------------------------
  // Destructor, constructors, assignment operators:

  KOKKOS_INLINE_FUNCTION
  ~View() { Impl::ViewTracking< traits >::decrement( m_ptr_on_device ); }

  KOKKOS_INLINE_FUNCTION
  View() : m_ptr_on_device(0)
    {
      traits::shape_type::assign(m_shape,0,0,0,0,0,0,0,0);
      stride_type::assign(m_stride,0);
    }

  KOKKOS_INLINE_FUNCTION
  View( const View & rhs ) : m_ptr_on_device(0) { assign_compatible_view( rhs ); }

  KOKKOS_INLINE_FUNCTION
  View & operator = ( const View & rhs ) { assign_compatible_view( rhs ); return *this ; }

  //------------------------------------
  // Construct or assign compatible view:

  template< class RT , class RL , class RD , class RM >
  KOKKOS_INLINE_FUNCTION
  View( const View<RT,RL,RD,RM,typename traits::specialize> & rhs )
    : m_ptr_on_device(0) { assign_compatible_view( rhs ); }

  template< class RT , class RL , class RD , class RM >
  KOKKOS_INLINE_FUNCTION
  View & operator = ( const View<RT,RL,RD,RM,typename traits::specialize> & rhs )
    { assign_compatible_view( rhs ); return *this ; }

  //------------------------------------
  // Allocation of a managed view with possible alignment padding.

  typedef Impl::if_c< traits::is_managed ,
                      std::string ,
                      Impl::ViewError::allocation_constructor_requires_managed >
   if_allocation_constructor ;

  explicit inline
  View( const typename if_allocation_constructor::type & label ,
        const size_t n0 = 0 ,
        const size_t n1 = 0 ,
        const size_t n2 = 0 ,
        const size_t n3 = 0 ,
        const size_t n4 = 0 ,
        const size_t n5 = 0 ,
        const size_t n6 = 0 ,
        const size_t n7 = 0 )
    : m_ptr_on_device(0)
    {
      typedef typename traits::device_type   device_type ;
      typedef typename traits::memory_space  memory_space ;
      typedef typename traits::shape_type    shape_type ;
      typedef typename traits::scalar_type   scalar_type ;

      shape_type ::assign( m_shape, n0, n1, n2, n3, n4, n5, n6, n7 );
      stride_type::assign_with_padding( m_stride , m_shape );

      m_ptr_on_device = (scalar_type *)
        memory_space::allocate( if_allocation_constructor::select( label ) ,
                                typeid(scalar_type) ,
                                sizeof(scalar_type) ,
                                Impl::capacity( m_shape , m_stride ) );

      Impl::ViewInitialize< device_type > init( *this );
    }

  explicit inline
  View( const AllocateWithoutInitializing & ,
        const typename if_allocation_constructor::type & label ,
        const size_t n0 = 0 ,
        const size_t n1 = 0 ,
        const size_t n2 = 0 ,
        const size_t n3 = 0 ,
        const size_t n4 = 0 ,
        const size_t n5 = 0 ,
        const size_t n6 = 0 ,
        const size_t n7 = 0 )
    : m_ptr_on_device(0)
    {
      typedef typename traits::device_type   device_type ;
      typedef typename traits::memory_space  memory_space ;
      typedef typename traits::shape_type    shape_type ;
      typedef typename traits::scalar_type   scalar_type ;

      shape_type ::assign( m_shape, n0, n1, n2, n3, n4, n5, n6, n7 );
      stride_type::assign_with_padding( m_stride , m_shape );

      m_ptr_on_device = (scalar_type *)
        memory_space::allocate( if_allocation_constructor::select( label ) ,
                                typeid(scalar_type) ,
                                sizeof(scalar_type) ,
                                Impl::capacity( m_shape , m_stride ) );
    }

  //------------------------------------
  // Assign an unmanaged View from pointer, can be called in functors.
  // No alignment padding is performed.

  typedef Impl::if_c< ! traits::is_managed ,
                      typename traits::scalar_type * ,
                      Impl::ViewError::user_pointer_constructor_requires_unmanaged >
    if_user_pointer_constructor ;

  View( typename if_user_pointer_constructor::type ptr ,
        const size_t n0 = 0 ,
        const size_t n1 = 0 ,
        const size_t n2 = 0 ,
        const size_t n3 = 0 ,
        const size_t n4 = 0 ,
        const size_t n5 = 0 ,
        const size_t n6 = 0 ,
        const size_t n7 = 0 )
    : m_ptr_on_device(0)
    {
      typedef typename traits::shape_type   shape_type ;
      typedef typename traits::scalar_type  scalar_type ;

      shape_type ::assign( m_shape, n0, n1, n2, n3, n4, n5, n6, n7 );
      stride_type::assign_no_padding( m_stride , m_shape );

      m_ptr_on_device = if_user_pointer_constructor::select( ptr );
    }

  //------------------------------------
  // Assign unmanaged View to portion of Device shared memory

  typedef Impl::if_c< ! traits::is_managed ,
                      typename traits::device_type ,
                      Impl::ViewError::device_shmem_constructor_requires_unmanaged >
      if_device_shmem_constructor ;

  explicit KOKKOS_INLINE_FUNCTION
  View( typename if_device_shmem_constructor::type & dev ,
        const unsigned n0 = 0 ,
        const unsigned n1 = 0 ,
        const unsigned n2 = 0 ,
        const unsigned n3 = 0 ,
        const unsigned n4 = 0 ,
        const unsigned n5 = 0 ,
        const unsigned n6 = 0 ,
        const unsigned n7 = 0 )
    : m_ptr_on_device(0)
    {
      typedef typename traits::shape_type   shape_type ;
      typedef typename traits::scalar_type  scalar_type ;

      enum { align = 8 };
      enum { mask  = align - 1 };

      shape_type::assign( m_shape, n0, n1, n2, n3, n4, n5, n6, n7 );
      stride_type::assign_no_padding( m_stride , m_shape );

      typedef Impl::if_c< ! traits::is_managed ,
                          scalar_type * ,
                          Impl::ViewError::device_shmem_constructor_requires_unmanaged >
        if_device_shmem_pointer ;

      // Select the first argument:
      m_ptr_on_device = if_device_shmem_pointer::select(
       (scalar_type *) dev.get_shmem( unsigned( sizeof(scalar_type) * Impl::capacity( m_shape , m_stride ) + unsigned(mask) ) & ~unsigned(mask) ) );
    }

  static inline
  unsigned shmem_size( const unsigned n0 = 0 ,
                       const unsigned n1 = 0 ,
                       const unsigned n2 = 0 ,
                       const unsigned n3 = 0 ,
                       const unsigned n4 = 0 ,
                       const unsigned n5 = 0 ,
                       const unsigned n6 = 0 ,
                       const unsigned n7 = 0 )
  {
    enum { align = 8 };
    enum { mask  = align - 1 };

    typedef typename traits::shape_type   shape_type ;
    typedef typename traits::scalar_type  scalar_type ;

    shape_type  shape ;
    stride_type stride ;
    
    traits::shape_type::assign( shape, n0, n1, n2, n3, n4, n5, n6, n7 );
    stride_type::assign_no_padding( stride , shape );

    return unsigned( sizeof(scalar_type) * Impl::capacity( shape , stride ) + unsigned(mask) ) & ~unsigned(mask) ;
  }

  //------------------------------------
  // Is not allocated

  KOKKOS_INLINE_FUNCTION
  bool is_null() const { return 0 == m_ptr_on_device ; }

  //------------------------------------
  // Operators for scalar (rank zero) views.

  typedef Impl::if_c< traits::rank == 0 ,
                      typename traits::scalar_type ,
                      Impl::ViewError::scalar_operator_called_from_non_scalar_view >
    if_scalar_operator ;

  KOKKOS_INLINE_FUNCTION
  const View & operator = ( const typename if_scalar_operator::type & rhs ) const
    {
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );
      *m_ptr_on_device = if_scalar_operator::select( rhs );
      return *this ;
    }

  KOKKOS_INLINE_FUNCTION
  operator typename if_scalar_operator::type & () const
    {
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );
      return if_scalar_operator::select( *m_ptr_on_device );
    }

  KOKKOS_INLINE_FUNCTION
  typename if_scalar_operator::type & operator()() const
    {
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );
      return if_scalar_operator::select( *m_ptr_on_device );
    }

  KOKKOS_INLINE_FUNCTION
  typename if_scalar_operator::type & operator*() const
    {
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );
      return if_scalar_operator::select( *m_ptr_on_device );
    }

  //------------------------------------
  // Array member access operators enabled if
  // (1) a zero value of all argument types are compile-time comparable to zero
  // (2) the rank matches the number of arguments
  // (3) the memory space is valid for the access
  //------------------------------------
  // LayoutLeft, rank 1:

  template< typename iType0 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & , traits, LayoutLeft, 1, iType0 >::type
    operator[] ( const iType0 & i0 ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_1( m_shape, i0 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i0 ];
    }

  template< typename iType0 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & , traits, LayoutLeft, 1, iType0 >::type
    operator() ( const iType0 & i0 ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_1( m_shape, i0 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i0 ];
    }

  template< typename iType0 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & , traits, LayoutLeft, 1, iType0 >::type
    at( const iType0 & i0 , const int , const int , const int ,
        const int , const int , const int , const int ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_1( m_shape, i0 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i0 ];
    }

  // LayoutLeft, rank 2:

  template< typename iType0 , typename iType1 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutLeft, 2, iType0, iType1 >::type
    operator() ( const iType0 & i0 , const iType1 & i1 ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_2( m_shape, i0,i1 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i0 + m_stride.value * i1 ];
    }

  template< typename iType0 , typename iType1 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutLeft, 2, iType0, iType1 >::type
    at( const iType0 & i0 , const iType1 & i1 , const int , const int ,
        const int , const int , const int , const int ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_2( m_shape, i0,i1 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i0 + m_stride.value * i1 ];
    }

  // LayoutLeft, rank 3:

  template< typename iType0 , typename iType1 , typename iType2 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutLeft, 3, iType0, iType1, iType2 >::type
    operator() ( const iType0 & i0 , const iType1 & i1 , const iType2 & i2 ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_3( m_shape, i0,i1,i2 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i0 + m_stride.value * (
                              i1 + m_shape.N1 * i2 ) ];
    }

  template< typename iType0 , typename iType1 , typename iType2 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutLeft, 3, iType0, iType1, iType2 >::type
    at( const iType0 & i0 , const iType1 & i1 , const iType2 & i2 , const int ,
        const int , const int , const int , const int ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_3( m_shape, i0,i1,i2 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i0 + m_stride.value * (
                              i1 + m_shape.N1 * i2 ) ];
    }

  // LayoutLeft, rank 4:

  template< typename iType0 , typename iType1 , typename iType2 , typename iType3 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutLeft, 4, iType0, iType1, iType2, iType3 >::type
    operator() ( const iType0 & i0 , const iType1 & i1 , const iType2 & i2 , const iType3 & i3 ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_4( m_shape, i0,i1,i2,i3 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i0 + m_stride.value * (
                              i1 + m_shape.N1 * (
                              i2 + m_shape.N2 * i3 )) ];
    }

  template< typename iType0 , typename iType1 , typename iType2 , typename iType3 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutLeft, 4, iType0, iType1, iType2, iType3 >::type
    at( const iType0 & i0 , const iType1 & i1 , const iType2 & i2 , const iType3 & i3 ,
        const int , const int , const int , const int ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_4( m_shape, i0,i1,i2,i3 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i0 + m_stride.value * (
                              i1 + m_shape.N1 * (
                              i2 + m_shape.N2 * i3 )) ];
    }

  // LayoutLeft, rank 5:

  template< typename iType0 , typename iType1 , typename iType2 , typename iType3 ,
            typename iType4 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutLeft, 5, iType0, iType1, iType2, iType3 , iType4 >::type
    operator() ( const iType0 & i0 , const iType1 & i1 , const iType2 & i2 , const iType3 & i3 ,
                 const iType4 & i4 ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_5( m_shape, i0,i1,i2,i3,i4 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i0 + m_stride.value * (
                              i1 + m_shape.N1 * (
                              i2 + m_shape.N2 * (
                              i3 + m_shape.N3 * i4 ))) ];
    }

  template< typename iType0 , typename iType1 , typename iType2 , typename iType3 ,
            typename iType4 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutLeft, 5, iType0, iType1, iType2, iType3 , iType4 >::type
    at( const iType0 & i0 , const iType1 & i1 , const iType2 & i2 , const iType3 & i3 ,
        const iType4 & i4 , const int , const int , const int ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_5( m_shape, i0,i1,i2,i3,i4 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i0 + m_stride.value * (
                              i1 + m_shape.N1 * (
                              i2 + m_shape.N2 * (
                              i3 + m_shape.N3 * i4 ))) ];
    }

  // LayoutLeft, rank 6:

  template< typename iType0 , typename iType1 , typename iType2 , typename iType3 ,
            typename iType4 , typename iType5 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutLeft, 6, iType0, iType1, iType2, iType3 , iType4, iType5 >::type
    operator() ( const iType0 & i0 , const iType1 & i1 , const iType2 & i2 , const iType3 & i3 ,
                 const iType4 & i4 , const iType5 & i5 ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_6( m_shape, i0,i1,i2,i3,i4,i5 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i0 + m_stride.value * (
                              i1 + m_shape.N1 * (
                              i2 + m_shape.N2 * (
                              i3 + m_shape.N3 * (
                              i4 + m_shape.N4 * i5 )))) ];
    }

  template< typename iType0 , typename iType1 , typename iType2 , typename iType3 ,
            typename iType4 , typename iType5 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutLeft, 6, iType0, iType1, iType2, iType3 , iType4, iType5 >::type
    at( const iType0 & i0 , const iType1 & i1 , const iType2 & i2 , const iType3 & i3 ,
        const iType4 & i4 , const iType5 & i5 , const int , const int ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_6( m_shape, i0,i1,i2,i3,i4,i5 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i0 + m_stride.value * (
                              i1 + m_shape.N1 * (
                              i2 + m_shape.N2 * (
                              i3 + m_shape.N3 * (
                              i4 + m_shape.N4 * i5 )))) ];
    }

  // LayoutLeft, rank 7:

  template< typename iType0 , typename iType1 , typename iType2 , typename iType3 ,
            typename iType4 , typename iType5 , typename iType6 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutLeft, 7, iType0, iType1, iType2, iType3 , iType4, iType5, iType6 >::type
    operator() ( const iType0 & i0 , const iType1 & i1 , const iType2 & i2 , const iType3 & i3 ,
                 const iType4 & i4 , const iType5 & i5 , const iType6 & i6 ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_7( m_shape, i0,i1,i2,i3,i4,i5,i6 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i0 + m_stride.value * (
                              i1 + m_shape.N1 * (
                              i2 + m_shape.N2 * (
                              i3 + m_shape.N3 * (
                              i4 + m_shape.N4 * (
                              i5 + m_shape.N5 * i6 ))))) ];
    }

  template< typename iType0 , typename iType1 , typename iType2 , typename iType3 ,
            typename iType4 , typename iType5 , typename iType6 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutLeft, 7, iType0, iType1, iType2, iType3 , iType4, iType5, iType6 >::type
    at( const iType0 & i0 , const iType1 & i1 , const iType2 & i2 , const iType3 & i3 ,
        const iType4 & i4 , const iType5 & i5 , const iType6 & i6 , const int ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_7( m_shape, i0,i1,i2,i3,i4,i5,i6 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i0 + m_stride.value * (
                              i1 + m_shape.N1 * (
                              i2 + m_shape.N2 * (
                              i3 + m_shape.N3 * (
                              i4 + m_shape.N4 * (
                              i5 + m_shape.N5 * i6 ))))) ];
    }

  // LayoutLeft, rank 8:

  template< typename iType0 , typename iType1 , typename iType2 , typename iType3 ,
            typename iType4 , typename iType5 , typename iType6 , typename iType7 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutLeft, 8, iType0, iType1, iType2, iType3 , iType4, iType5, iType6, iType7 >::type
    operator() ( const iType0 & i0 , const iType1 & i1 , const iType2 & i2 , const iType3 & i3 ,
                 const iType4 & i4 , const iType5 & i5 , const iType6 & i6 , const iType7 & i7 ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_8( m_shape, i0,i1,i2,i3,i4,i5,i6,i7 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i0 + m_stride.value * (
                              i1 + m_shape.N1 * (
                              i2 + m_shape.N2 * (
                              i3 + m_shape.N3 * (
                              i4 + m_shape.N4 * (
                              i5 + m_shape.N5 * (
                              i6 + m_shape.N6 * i7 )))))) ];
    }

  template< typename iType0 , typename iType1 , typename iType2 , typename iType3 ,
            typename iType4 , typename iType5 , typename iType6 , typename iType7 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutLeft, 8, iType0, iType1, iType2, iType3 , iType4, iType5, iType6, iType7 >::type
    at( const iType0 & i0 , const iType1 & i1 , const iType2 & i2 , const iType3 & i3 ,
        const iType4 & i4 , const iType5 & i5 , const iType6 & i6 , const iType7 & i7 ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_8( m_shape, i0,i1,i2,i3,i4,i5,i6,i7 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i0 + m_stride.value * (
                              i1 + m_shape.N1 * (
                              i2 + m_shape.N2 * (
                              i3 + m_shape.N3 * (
                              i4 + m_shape.N4 * (
                              i5 + m_shape.N5 * (
                              i6 + m_shape.N6 * i7 )))))) ];
    }

  //------------------------------------
  // LayoutRight, rank 1:

  template< typename iType0 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & , traits, LayoutRight, 1, iType0 >::type
    operator[] ( const iType0 & i0 ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_1( m_shape, i0 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i0 ];
    }

  template< typename iType0 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & , traits, LayoutRight, 1, iType0 >::type
    operator() ( const iType0 & i0 ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_1( m_shape, i0 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i0 ];
    }

  template< typename iType0 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & , traits, LayoutRight, 1, iType0 >::type
    at( const iType0 & i0 , const int , const int , const int ,
        const int , const int , const int , const int ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_1( m_shape, i0 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i0 ];
    }

  // LayoutRight, rank 2:

  template< typename iType0 , typename iType1 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutRight, 2, iType0, iType1 >::type
    operator() ( const iType0 & i0 , const iType1 & i1 ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_2( m_shape, i0,i1 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i1 + i0 * m_stride.value ];
    }

  template< typename iType0 , typename iType1 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutRight, 2, iType0, iType1 >::type
    at( const iType0 & i0 , const iType1 & i1 , const int , const int ,
        const int , const int , const int , const int ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_2( m_shape, i0,i1 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i1 + i0 * m_stride.value ];
    }

  // LayoutRight, rank 3:

  template< typename iType0 , typename iType1 , typename iType2 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutRight, 3, iType0, iType1, iType2 >::type
    operator() ( const iType0 & i0 , const iType1 & i1 , const iType2 & i2 ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_3( m_shape, i0,i1,i2 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i2 + m_shape.N2 * ( i1 ) + i0 * m_stride.value ];
    }

  template< typename iType0 , typename iType1 , typename iType2 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutRight, 3, iType0, iType1, iType2 >::type
    at( const iType0 & i0 , const iType1 & i1 , const iType2 & i2 , const int ,
        const int , const int , const int , const int ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_3( m_shape, i0,i1,i2 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i2 + m_shape.N2 * ( i1 ) + i0 * m_stride.value ];
    }

  // LayoutRight, rank 4:

  template< typename iType0 , typename iType1 , typename iType2 , typename iType3 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutRight, 4, iType0, iType1, iType2, iType3 >::type
    operator() ( const iType0 & i0 , const iType1 & i1 , const iType2 & i2 , const iType3 & i3 ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_4( m_shape, i0,i1,i2,i3 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i3 + m_shape.N3 * (
                              i2 + m_shape.N2 * (
                              i1 )) + i0 * m_stride.value ];
    }

  template< typename iType0 , typename iType1 , typename iType2 , typename iType3 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutRight, 4, iType0, iType1, iType2, iType3 >::type
    at( const iType0 & i0 , const iType1 & i1 , const iType2 & i2 , const iType3 & i3 ,
        const int , const int , const int , const int ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_4( m_shape, i0,i1,i2,i3 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i3 + m_shape.N3 * (
                              i2 + m_shape.N2 * (
                              i1 )) + i0 * m_stride.value ];
    }

  // LayoutRight, rank 5:

  template< typename iType0 , typename iType1 , typename iType2 , typename iType3 ,
            typename iType4 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutRight, 5, iType0, iType1, iType2, iType3, iType4 >::type
    operator() ( const iType0 & i0 , const iType1 & i1 , const iType2 & i2 , const iType3 & i3 ,
                 const iType4 & i4 ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_5( m_shape, i0,i1,i2,i3,i4 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i4 + m_shape.N4 * (
                              i3 + m_shape.N3 * (
                              i2 + m_shape.N2 * (
                              i1 ))) + i0 * m_stride.value ];
    }

  template< typename iType0 , typename iType1 , typename iType2 , typename iType3 ,
            typename iType4 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutRight, 5, iType0, iType1, iType2, iType3, iType4 >::type
    at( const iType0 & i0 , const iType1 & i1 , const iType2 & i2 , const iType3 & i3 ,
        const iType4 & i4 , const int , const int , const int ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_5( m_shape, i0,i1,i2,i3,i4 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i4 + m_shape.N4 * (
                              i3 + m_shape.N3 * (
                              i2 + m_shape.N2 * (
                              i1 ))) + i0 * m_stride.value ];
    }

  // LayoutRight, rank 6:

  template< typename iType0 , typename iType1 , typename iType2 , typename iType3 ,
            typename iType4 , typename iType5 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutRight, 6, iType0, iType1, iType2, iType3, iType4, iType5 >::type
    operator() ( const iType0 & i0 , const iType1 & i1 , const iType2 & i2 , const iType3 & i3 ,
                 const iType4 & i4 , const iType5 & i5 ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_6( m_shape, i0,i1,i2,i3,i4,i5 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i5 + m_shape.N5 * (
                              i4 + m_shape.N4 * (
                              i3 + m_shape.N3 * (
                              i2 + m_shape.N2 * (
                              i1 )))) + i0 * m_stride.value ];
    }

  template< typename iType0 , typename iType1 , typename iType2 , typename iType3 ,
            typename iType4 , typename iType5 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutRight, 6, iType0, iType1, iType2, iType3, iType4, iType5 >::type
    at( const iType0 & i0 , const iType1 & i1 , const iType2 & i2 , const iType3 & i3 ,
        const iType4 & i4 , const iType5 & i5 , const int , const int ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_6( m_shape, i0,i1,i2,i3,i4,i5 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i5 + m_shape.N5 * (
                              i4 + m_shape.N4 * (
                              i3 + m_shape.N3 * (
                              i2 + m_shape.N2 * (
                              i1 )))) + i0 * m_stride.value ];
    }

  // LayoutRight, rank 7:

  template< typename iType0 , typename iType1 , typename iType2 , typename iType3 ,
            typename iType4 , typename iType5 , typename iType6 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutRight, 7, iType0, iType1, iType2, iType3, iType4, iType5, iType6 >::type
    operator() ( const iType0 & i0 , const iType1 & i1 , const iType2 & i2 , const iType3 & i3 ,
                 const iType4 & i4 , const iType5 & i5 , const iType6 & i6 ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_7( m_shape, i0,i1,i2,i3,i4,i5,i6 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i6 + m_shape.N6 * (
                              i5 + m_shape.N5 * (
                              i4 + m_shape.N4 * (
                              i3 + m_shape.N3 * (
                              i2 + m_shape.N2 * (
                              i1 ))))) + i0 * m_stride.value ];
    }

  template< typename iType0 , typename iType1 , typename iType2 , typename iType3 ,
            typename iType4 , typename iType5 , typename iType6 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutRight, 7, iType0, iType1, iType2, iType3, iType4, iType5, iType6 >::type
    at( const iType0 & i0 , const iType1 & i1 , const iType2 & i2 , const iType3 & i3 ,
        const iType4 & i4 , const iType5 & i5 , const iType6 & i6 , const int ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_7( m_shape, i0,i1,i2,i3,i4,i5,i6 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i6 + m_shape.N6 * (
                              i5 + m_shape.N5 * (
                              i4 + m_shape.N4 * (
                              i3 + m_shape.N3 * (
                              i2 + m_shape.N2 * (
                              i1 ))))) + i0 * m_stride.value ];
    }

  // LayoutRight, rank 8:

  template< typename iType0 , typename iType1 , typename iType2 , typename iType3 ,
            typename iType4 , typename iType5 , typename iType6 , typename iType7 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutRight, 8, iType0, iType1, iType2, iType3, iType4, iType5, iType6, iType7 >::type
    operator() ( const iType0 & i0 , const iType1 & i1 , const iType2 & i2 , const iType3 & i3 ,
                 const iType4 & i4 , const iType5 & i5 , const iType6 & i6 , const iType7 & i7 ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_8( m_shape, i0,i1,i2,i3,i4,i5,i6,i7 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i7 + m_shape.N7 * (
                              i6 + m_shape.N6 * (
                              i5 + m_shape.N5 * (
                              i4 + m_shape.N4 * (
                              i3 + m_shape.N3 * (
                              i2 + m_shape.N2 * (
                              i1 )))))) + i0 * m_stride.value ];
    }

  template< typename iType0 , typename iType1 , typename iType2 , typename iType3 ,
            typename iType4 , typename iType5 , typename iType6 , typename iType7 >
  KOKKOS_INLINE_FUNCTION
  typename Impl::ViewEnableArrayOper< typename traits::scalar_type & ,
                                      traits, LayoutRight, 8, iType0, iType1, iType2, iType3, iType4, iType5, iType6, iType7 >::type
    at( const iType0 & i0 , const iType1 & i1 , const iType2 & i2 , const iType3 & i3 ,
        const iType4 & i4 , const iType5 & i5 , const iType6 & i6 , const iType7 & i7 ) const
    {
      KOKKOS_ASSERT_SHAPE_BOUNDS_8( m_shape, i0,i1,i2,i3,i4,i5,i6,i7 );
      KOKKOS_RESTRICT_EXECUTION_TO_DATA( typename traits::memory_space , m_ptr_on_device );

      return m_ptr_on_device[ i7 + m_shape.N7 * (
                              i6 + m_shape.N6 * (
                              i5 + m_shape.N5 * (
                              i4 + m_shape.N4 * (
                              i3 + m_shape.N3 * (
                              i2 + m_shape.N2 * (
                              i1 )))))) + i0 * m_stride.value ];
    }

  //------------------------------------
  // Access to the underlying contiguous storage of this view specialization.
  // These methods are specific to specialization of a view.

  KOKKOS_INLINE_FUNCTION
  typename traits::scalar_type * ptr_on_device() const { return m_ptr_on_device ; }

  // Stride of physical storage, dimensioned to at least Rank
  template< typename iType >
  KOKKOS_INLINE_FUNCTION
  void stride( iType * const s ) const
  { Impl::stride( s , m_shape , m_stride ); }

  // Count of contiguously allocated data members including padding.
  KOKKOS_INLINE_FUNCTION
  typename traits::size_type capacity() const
  { return Impl::capacity( m_shape , m_stride ); }
};

} /* namespace Kokkos */

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

template< class LT , class LL , class LD , class LM , class LS ,
          class RT , class RL , class RD , class RM , class RS >
KOKKOS_INLINE_FUNCTION
typename Impl::enable_if<( Impl::is_same< LS , RS >::value ), bool >::type
operator == ( const View<LT,LL,LD,LM,LS> & lhs ,
              const View<RT,RL,RD,RM,RS> & rhs )
{
  // Same data, layout, dimensions
  typedef ViewTraits<LT,LL,LD,LM> lhs_traits ;
  typedef ViewTraits<RT,RL,RD,RM> rhs_traits ;

  return
    Impl::is_same< typename lhs_traits::const_data_type ,
                   typename rhs_traits::const_data_type >::value &&
    Impl::is_same< typename lhs_traits::array_layout ,
                   typename rhs_traits::array_layout >::value &&
    Impl::is_same< typename lhs_traits::memory_space ,
                   typename rhs_traits::memory_space >::value &&
    Impl::is_same< typename lhs_traits::specialize ,
                   typename rhs_traits::specialize >::value &&
    lhs.ptr_on_device() == rhs.ptr_on_device() &&
    lhs.shape()         == rhs.shape() ;
}

template< class LT , class LL , class LD , class LM , class LS ,
          class RT , class RL , class RD , class RM , class RS >
KOKKOS_INLINE_FUNCTION
bool operator != ( const View<LT,LL,LD,LM,LS> & lhs ,
                   const View<RT,RL,RD,RM,RS> & rhs )
{
  return ! operator==( lhs , rhs );
}

//----------------------------------------------------------------------------


} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

//----------------------------------------------------------------------------
/** \brief  Deep copy a value into a view.
 */
template< class DT , class DL , class DD , class DM , class DS >
inline
void deep_copy( const View<DT,DL,DD,DM,DS> & dst ,
                typename Impl::enable_if<(
                  Impl::is_same< typename ViewTraits<DT,DL,DD,DM>::non_const_scalar_type ,
                                 typename ViewTraits<DT,DL,DD,DM>::scalar_type >::value
                ), typename ViewTraits<DT,DL,DD,DM>::const_scalar_type >::type & value )
{
  Impl::ViewFill< View<DT,DL,DD,DM,DS> >( dst , value );
}

template< class ST , class SL , class SD , class SM , class SS >
inline
typename Impl::enable_if<( ViewTraits<ST,SL,SD,SM>::rank == 0 )>::type
deep_copy( ST & dst , const View<ST,SL,SD,SM,SS> & src )
{
  typedef  ViewTraits<ST,SL,SD,SM>  src_traits ;
  typedef typename src_traits::memory_space  src_memory_space ;
  Impl::DeepCopy< HostSpace , src_memory_space >( & dst , src.ptr_on_device() , sizeof(ST) );
}

//----------------------------------------------------------------------------
/** \brief  A deep copy between views of the same specialization, compatible type,
 *          same rank, same layout are handled by that specialization.
 */
template< class DT , class DL , class DD , class DM , class DS ,
          class ST , class SL , class SD , class SM , class SS >
inline
void deep_copy( const View<DT,DL,DD,DM,DS> & dst ,
                const View<ST,SL,SD,SM,SS> & src ,
                typename Impl::enable_if<(
                  Impl::is_same< typename ViewTraits<DT,DL,DD,DM>::scalar_type ,
                                 typename ViewTraits<ST,SL,SD,SM>::non_const_scalar_type >::value
                  &&
                  Impl::is_same< typename ViewTraits<DT,DL,DD,DM>::array_layout ,
                                 typename ViewTraits<ST,SL,SD,SM>::array_layout >::value
                  &&
                  ( unsigned(ViewTraits<DT,DL,DD,DM>::rank) == unsigned(ViewTraits<ST,SL,SD,SM>::rank) )
                )>::type * = 0 )
{
  typedef  ViewTraits<DT,DL,DD,DM>  dst_traits ;
  typedef  ViewTraits<ST,SL,SD,SM>  src_traits ;

  typedef typename dst_traits::memory_space  dst_memory_space ;
  typedef typename src_traits::memory_space  src_memory_space ;

  if ( dst.ptr_on_device() != src.ptr_on_device() ) {

    Impl::assert_shapes_are_equal( dst.shape() , src.shape() );

    const size_t nbytes = sizeof(typename dst_traits::scalar_type) * dst.capacity();

    Impl::DeepCopy< dst_memory_space , src_memory_space >( dst.ptr_on_device() , src.ptr_on_device() , nbytes );
  }
}


/** \brief Deep copy equal dimension arrays in the host space which
 *         have different layouts or specializations.
 */
template< class DT , class DL , class DD , class DM , class DS ,
          class ST , class SL ,            class SM , class SS >
inline
void deep_copy( const View< DT, DL, DD, DM, DS> & dst ,
                const View< ST, SL, DD, SM, SS> & src ,
                const typename Impl::enable_if<(
                  // Destination is not constant:
                  Impl::is_same< typename ViewTraits<DT,DL,DD,DM>::value_type ,
                                 typename ViewTraits<DT,DL,DD,DM>::non_const_value_type >::value
                  &&
                  // Same rank
                  ( unsigned( ViewTraits<DT,DL,DD,DM>::rank ) ==
                    unsigned( ViewTraits<ST,SL,DD,SM>::rank ) )
                  &&
                  // Different layout or different specialization:
                  ( ( ! Impl::is_same< typename DL::array_layout ,
                                       typename SL::array_layout >::value )
                    ||
                    ( ! Impl::is_same< DS , SS >::value )
                  )
                )>::type * = 0 )
{
  typedef View< DT, DL, DD, DM, DS> dst_type ;
  typedef View< ST, SL, DD, SM, SS> src_type ;

  assert_shapes_equal_dimension( dst.shape() , src.shape() );

  Impl::ViewRemap< dst_type , src_type >( dst , src );
}

//----------------------------------------------------------------------------

template< class T , class L , class D , class M , class S >
typename Impl::enable_if<(
    View<T,L,D,M,S>::is_managed 
  ), typename View<T,L,D,M,S>::HostMirror >::type
inline
create_mirror( const View<T,L,D,M,S> & src )
{
  typedef View<T,L,D,M,S>                  view_type ;
  typedef typename view_type::HostMirror    host_view_type ;
  typedef typename view_type::memory_space  memory_space ;

  // 'view' is managed therefore we can allocate a
  // compatible host_view through the ordinary constructor.
  
  std::string label = memory_space::query_label( src.ptr_on_device() );
  label.append("_mirror");

  return host_view_type( label ,
                         src.dimension_0() ,
                         src.dimension_1() ,
                         src.dimension_2() ,
                         src.dimension_3() ,
                         src.dimension_4() ,
                         src.dimension_5() ,
                         src.dimension_6() ,
                         src.dimension_7() );
}

template< class T , class L , class D , class M , class S >
typename Impl::enable_if<(
    View<T,L,D,M,S>::is_managed &&
    Impl::ViewAssignable< typename View<T,L,D,M,S>::HostMirror , View<T,L,D,M,S> >::value
  ), typename View<T,L,D,M,S>::HostMirror >::type
inline
create_mirror_view( const View<T,L,D,M,S> & src )
{
  return src ;
}

template< class T , class L , class D , class M , class S >
typename Impl::enable_if<(
    View<T,L,D,M,S>::is_managed &&
    ! Impl::ViewAssignable< typename View<T,L,D,M,S>::HostMirror , View<T,L,D,M,S> >::value
  ), typename View<T,L,D,M,S>::HostMirror >::type
inline
create_mirror_view( const View<T,L,D,M,S> & src )
{
  return create_mirror( src );
}

//----------------------------------------------------------------------------

/** \brief  Resize a view with copying old data to new data at the corresponding indices. */
template< class T , class L , class D , class M , class S >
inline
void resize( View<T,L,D,M,S> & v ,
             const typename Impl::enable_if< ViewTraits<T,L,D,M>::is_managed , size_t >::type n0 ,
             const size_t n1 = 0 ,
             const size_t n2 = 0 ,
             const size_t n3 = 0 ,
             const size_t n4 = 0 ,
             const size_t n5 = 0 ,
             const size_t n6 = 0 ,
             const size_t n7 = 0 )
{
  typedef View<T,L,D,M,S> view_type ;
  typedef typename view_type::memory_space memory_space ;

  const std::string label = memory_space::query_label( v.ptr_on_device() );

  view_type v_resized( label, n0, n1, n2, n3, n4, n5, n6, n7 );

  Impl::ViewRemap< view_type , view_type >( v_resized , v );

  v = v_resized ;
}

/** \brief  Reallocate a view without copying old data to new data */
template< class T , class L , class D , class M , class S >
inline
void realloc( View<T,L,D,M,S> & v ,
              const typename Impl::enable_if< ViewTraits<T,L,D,M>::is_managed , size_t >::type n0 ,
              const size_t n1 = 0 ,
              const size_t n2 = 0 ,
              const size_t n3 = 0 ,
              const size_t n4 = 0 ,
              const size_t n5 = 0 ,
              const size_t n6 = 0 ,
              const size_t n7 = 0 )
{
  typedef View<T,L,D,M,S> view_type ;
  typedef typename view_type::memory_space memory_space ;

  // Query the current label and reuse it.
  const std::string label = memory_space::query_label( v.ptr_on_device() );

  v = view_type(); // deallocate first, if the only view to memory.
  v = view_type( label, n0, n1, n2, n3, n4, n5, n6, n7 );
}

} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

struct ALL { KOKKOS_INLINE_FUNCTION ALL(){} };

template< class DstViewType ,
          class T , class L , class D , class M , class S ,
          class ArgType0 >
KOKKOS_INLINE_FUNCTION
DstViewType
subview( const View<T,L,D,M,S> & src ,
         const ArgType0 & arg0 )
{
  DstViewType dst ;

  Impl::ViewAssignment<typename DstViewType::specialize,S>( dst , src , arg0 );

  return dst ;
}

template< class DstViewType ,
          class T , class L , class D , class M , class S ,
          class ArgType0 , class ArgType1 >
KOKKOS_INLINE_FUNCTION
DstViewType
subview( const View<T,L,D,M,S> & src ,
         const ArgType0 & arg0 ,
         const ArgType1 & arg1 )
{
  DstViewType dst ;

  Impl::ViewAssignment<typename DstViewType::specialize,S>( dst, src, arg0, arg1 );

  return dst ;
}

template< class DstViewType ,
          class T , class L , class D , class M , class S ,
          class ArgType0 , class ArgType1 , class ArgType2 >
KOKKOS_INLINE_FUNCTION
DstViewType
subview( const View<T,L,D,M,S> & src ,
         const ArgType0 & arg0 ,
         const ArgType1 & arg1 ,
         const ArgType2 & arg2 )
{
  DstViewType dst ;

  Impl::ViewAssignment<typename DstViewType::specialize,S>( dst, src, arg0, arg1, arg2 );

  return dst ;
}

template< class DstViewType ,
          class T , class L , class D , class M , class S ,
          class ArgType0 , class ArgType1 , class ArgType2 , class ArgType3 >
KOKKOS_INLINE_FUNCTION
DstViewType
subview( const View<T,L,D,M,S> & src ,
         const ArgType0 & arg0 ,
         const ArgType1 & arg1 ,
         const ArgType2 & arg2 ,
         const ArgType3 & arg3 )
{
  DstViewType dst ;

  Impl::ViewAssignment<typename DstViewType::specialize,S>( dst, src, arg0, arg1, arg2, arg3 );

  return dst ;
}

template< class DstViewType ,
          class T , class L , class D , class M , class S ,
          class ArgType0 , class ArgType1 , class ArgType2 , class ArgType3 ,
          class ArgType4 >
KOKKOS_INLINE_FUNCTION
DstViewType
subview( const View<T,L,D,M,S> & src ,
         const ArgType0 & arg0 ,
         const ArgType1 & arg1 ,
         const ArgType2 & arg2 ,
         const ArgType3 & arg3 ,
         const ArgType4 & arg4 )
{
  DstViewType dst ;

  Impl::ViewAssignment<typename DstViewType::specialize,S>( dst, src, arg0, arg1, arg2, arg3, arg4 );

  return dst ;
}

template< class DstViewType ,
          class T , class L , class D , class M , class S ,
          class ArgType0 , class ArgType1 , class ArgType2 , class ArgType3 ,
          class ArgType4 , class ArgType5 >
KOKKOS_INLINE_FUNCTION
DstViewType
subview( const View<T,L,D,M,S> & src ,
         const ArgType0 & arg0 ,
         const ArgType1 & arg1 ,
         const ArgType2 & arg2 ,
         const ArgType3 & arg3 ,
         const ArgType4 & arg4 ,
         const ArgType5 & arg5 )
{
  DstViewType dst ;

  Impl::ViewAssignment<typename DstViewType::specialize,S>( dst, src, arg0, arg1, arg2, arg3, arg4, arg5 );

  return dst ;
}

template< class DstViewType ,
          class T , class L , class D , class M , class S ,
          class ArgType0 , class ArgType1 , class ArgType2 , class ArgType3 ,
          class ArgType4 , class ArgType5 , class ArgType6 >
KOKKOS_INLINE_FUNCTION
DstViewType
subview( const View<T,L,D,M,S> & src ,
         const ArgType0 & arg0 ,
         const ArgType1 & arg1 ,
         const ArgType2 & arg2 ,
         const ArgType3 & arg3 ,
         const ArgType4 & arg4 ,
         const ArgType5 & arg5 ,
         const ArgType6 & arg6 )
{
  DstViewType dst ;

  Impl::ViewAssignment<typename DstViewType::specialize,S>( dst, src, arg0, arg1, arg2, arg3, arg4, arg5, arg6 );

  return dst ;
}

template< class DstViewType ,
          class T , class L , class D , class M , class S ,
          class ArgType0 , class ArgType1 , class ArgType2 , class ArgType3 ,
          class ArgType4 , class ArgType5 , class ArgType6 , class ArgType7 >
KOKKOS_INLINE_FUNCTION
DstViewType
subview( const View<T,L,D,M,S> & src ,
         const ArgType0 & arg0 ,
         const ArgType1 & arg1 ,
         const ArgType2 & arg2 ,
         const ArgType3 & arg3 ,
         const ArgType4 & arg4 ,
         const ArgType5 & arg5 ,
         const ArgType6 & arg6 ,
         const ArgType7 & arg7 )
{
  DstViewType dst ;

  Impl::ViewAssignment<typename DstViewType::specialize,S>( dst, src, arg0, arg1, arg2, arg3, arg4, arg5, arg6, arg7 );

  return dst ;
}

} // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#include <impl/Kokkos_ViewDefault.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

#endif

