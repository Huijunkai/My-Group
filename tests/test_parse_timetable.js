const fs = require('fs');
const path = require('path');
const { parseTimetable } = require('../src/parser');

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

function main() {
  const htmlPath = path.join(__dirname, 'fixtures', 'timetable_sample.html');
  const html = fs.readFileSync(htmlPath, 'utf-8');
  const courses = parseTimetable(html);

  console.log('解析到课程条数:', courses.length);
  assert(Array.isArray(courses), 'parseTimetable 应返回数组');
  assert(courses.length > 0, '课程条数不应为 0（通常意味着 <br/> 未被正确解析）');

  // 因为实现是“按周拆分存储”，这里应至少包含 1-8 周和 9-11 周
  const hasWeek1 = courses.some(c => c.week === 1 && c.dayOfWeek === '星期一' && c.period === '01-02节');
  const hasWeek9 = courses.some(c => c.week === 9 && c.dayOfWeek === '星期一' && c.period === '01-02节');
  assert(hasWeek1, '应包含 第1周 星期一 01-02节 的课程');
  assert(hasWeek9, '应包含 第9周 星期一 01-02节 的课程');

  // 简单打印前几条
  console.log('前3条示例:');
  console.log(courses.slice(0, 3));
}

main();

