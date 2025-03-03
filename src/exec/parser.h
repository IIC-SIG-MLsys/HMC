/**
 * @copyright Copyright (c) 2025, SDU spgroup Holding Limited
 */
#ifndef HDDT_EXEC_PARSER_H
#define HDDT_EXEC_PARSER_H

#include <hddt.h>
#include <forward_list>

#include "../utils/log.h"

namespace hddt {

class TaskMeta;

/* To get taskLists of this rank */
class Parser {
public:
  virtual uint32_t getParallelism() = 0; // how many executors we need at same time.
  virtual std::forward_list<TaskMeta>* getTaskList(uint32_t index) = 0; 
};

class ProtoParser: public Parser {
public:
  ProtoParser(std::string Proto_file, uint32_t node_rank);
  
  uint32_t getParallelism();
  std::forward_list<TaskMeta>* getTaskList(uint32_t index);

  ~ProtoParser();

private:
  std::vector<std::forward_list<TaskMeta>> TaskLists; // All TaskLists of the node
};

}

#endif
