#include <memory>
#include <iostream>
#include <regex>
#include <utility>
#include <string>
#include <cstdlib>
#include <CL/cl.h>


void check_success(cl_int returncode){
  if(returncode != CL_SUCCESS){
    std::cerr << "OpenCL Runtime error: " << returncode << std::endl;
    exit (-1);
  }
}


using version_t = std::pair<int, int>;

version_t parse_version_str(std::string const& str){
  // OpenCL API returns:
  //   OpenCL<space><major_version.minor_version><space><platform-specific information>
  std::regex const version_str_regex(R"(OpenCL (\d+)\.(\d+))");
  std::smatch match;

  if (!std::regex_search(str, match, version_str_regex)){
    std::cerr << "Illegal Version String" << std::endl;
    exit (1);
  }

  int const major_version = std::stoi(match[1].str());
  int const minor_version = std::stoi(match[2].str());
  return version_t(major_version, minor_version);
}


int main(int argc, char const** argv){
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <OpenCL Version>\nExample: " << argv[0] << " 1.2" << std::endl;
    exit (1);
  }
  auto const capable_version_str = std::string("OpenCL ") + argv[1];
  version_t const capable_version = parse_version_str(capable_version_str);

  // OpenCL C Version
#ifndef CL_VERSION_1_2
#define CL_VERSION_1_2 0
#endif
  std::cout << "OpenCL C Version:";
  if (CL_VERSION_1_2 == 1){
    std::cout << " OK" << std::endl;
  }else{
    std::cout << " must be >= " << argv[1] << std::endl;
    return 1;
  }


  // Platform Version
  cl_uint num_platforms;
  cl_int returncode = clGetPlatformIDs(0, NULL, &num_platforms);
  check_success(returncode);

  cl_platform_id platform;
  returncode = clGetPlatformIDs(1, &platform, &num_platforms);
  check_success(returncode);

  size_t param_value_size;
  returncode = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 0, NULL, &param_value_size);
  check_success(returncode);

  auto platform_version_str = std::unique_ptr<char[]>(new char[param_value_size]);
  returncode = clGetPlatformInfo(platform, CL_PLATFORM_VERSION, param_value_size, platform_version_str.get(), &param_value_size);
  check_success(returncode);

  auto const platform_version = parse_version_str(std::string(platform_version_str.get()));
  std::cout << "Platform Version: " << platform_version.first << "." << platform_version.second;
  if (platform_version >= capable_version){
    std::cout << " OK" << std::endl;
  }else{
    std::cout << " must be >= " << argv[1] << std::endl;
    return 1;
  }


  // Device Version
  cl_uint num_devices;
  returncode = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, 0, NULL, &num_devices);
  check_success(returncode);

  auto devices = std::unique_ptr<cl_device_id[]>(new cl_device_id[num_devices]);
  returncode = clGetDeviceIDs(platform, CL_DEVICE_TYPE_DEFAULT, num_devices, devices.get(), &num_devices);
  check_success(returncode);

  returncode = clGetDeviceInfo(devices[0], CL_DEVICE_VERSION, 0, NULL, &param_value_size);
  check_success(returncode);

  auto device_version_str = std::unique_ptr<char[]>(new char[param_value_size]);
  returncode = clGetDeviceInfo(devices[0], CL_DEVICE_VERSION, param_value_size, device_version_str.get(), &param_value_size);
  check_success(returncode);

  auto const device_version = parse_version_str(std::string(device_version_str.get()));
  std::cout << "Device Version: " << device_version.first << "." << device_version.second;
  if (device_version >= capable_version){
    std::cout << " OK" << std::endl;
  }else{
    std::cout << " must be >= " << argv[1] << std::endl;
    return 1;
  }


  return 0;
}
