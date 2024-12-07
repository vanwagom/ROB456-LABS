// Generated by gencpp from file lab2/NavTargetResult.msg
// DO NOT EDIT!


#ifndef LAB2_MESSAGE_NAVTARGETRESULT_H
#define LAB2_MESSAGE_NAVTARGETRESULT_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>

#include <std_msgs/Bool.h>

namespace lab2
{
template <class ContainerAllocator>
struct NavTargetResult_
{
  typedef NavTargetResult_<ContainerAllocator> Type;

  NavTargetResult_()
    : success()  {
    }
  NavTargetResult_(const ContainerAllocator& _alloc)
    : success(_alloc)  {
  (void)_alloc;
    }



   typedef  ::std_msgs::Bool_<ContainerAllocator>  _success_type;
  _success_type success;





  typedef boost::shared_ptr< ::lab2::NavTargetResult_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::lab2::NavTargetResult_<ContainerAllocator> const> ConstPtr;

}; // struct NavTargetResult_

typedef ::lab2::NavTargetResult_<std::allocator<void> > NavTargetResult;

typedef boost::shared_ptr< ::lab2::NavTargetResult > NavTargetResultPtr;
typedef boost::shared_ptr< ::lab2::NavTargetResult const> NavTargetResultConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::lab2::NavTargetResult_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::lab2::NavTargetResult_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::lab2::NavTargetResult_<ContainerAllocator1> & lhs, const ::lab2::NavTargetResult_<ContainerAllocator2> & rhs)
{
  return lhs.success == rhs.success;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::lab2::NavTargetResult_<ContainerAllocator1> & lhs, const ::lab2::NavTargetResult_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace lab2

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::lab2::NavTargetResult_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::lab2::NavTargetResult_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::lab2::NavTargetResult_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::lab2::NavTargetResult_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::lab2::NavTargetResult_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::lab2::NavTargetResult_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::lab2::NavTargetResult_<ContainerAllocator> >
{
  static const char* value()
  {
    return "5f31cb2e813cfb0e488c574c3b5d9dbe";
  }

  static const char* value(const ::lab2::NavTargetResult_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x5f31cb2e813cfb0eULL;
  static const uint64_t static_value2 = 0x488c574c3b5d9dbeULL;
};

template<class ContainerAllocator>
struct DataType< ::lab2::NavTargetResult_<ContainerAllocator> >
{
  static const char* value()
  {
    return "lab2/NavTargetResult";
  }

  static const char* value(const ::lab2::NavTargetResult_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::lab2::NavTargetResult_<ContainerAllocator> >
{
  static const char* value()
  {
    return "# ====== DO NOT MODIFY! AUTOGENERATED FROM AN ACTION DEFINITION ======\n"
"std_msgs/Bool success\n"
"\n"
"================================================================================\n"
"MSG: std_msgs/Bool\n"
"bool data\n"
;
  }

  static const char* value(const ::lab2::NavTargetResult_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::lab2::NavTargetResult_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.success);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct NavTargetResult_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::lab2::NavTargetResult_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::lab2::NavTargetResult_<ContainerAllocator>& v)
  {
    s << indent << "success: ";
    s << std::endl;
    Printer< ::std_msgs::Bool_<ContainerAllocator> >::stream(s, indent + "  ", v.success);
  }
};

} // namespace message_operations
} // namespace ros

#endif // LAB2_MESSAGE_NAVTARGETRESULT_H
