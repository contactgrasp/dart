//
// Created by samarth on 7/17/18.
// python interface to the DART tracker
//

#include <boost/python.hpp>
#include "grasp_analyzer.hpp"

using namespace std;

BOOST_PYTHON_MODULE(libpydart)
{
  using namespace boost::python;
  class_<GraspAnalyser, boost::noncopyable>("GraspAnalyser",
      init<const string &, const string &, const string &,
           optional<bool>, optional<bool>, optional<const string &> >()).
      def("set_params", &GraspAnalyser::set_params).
      def("analyze_grasps", &GraspAnalyser::analyze_grasps).
      def("close", &GraspAnalyser::close);
}
