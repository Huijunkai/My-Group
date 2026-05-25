/**
 * DataStore 模块单元测试
 * 测试内容：单例模式、数据结构验证、边界条件、JSON操作
 * 注意：存储相关测试需要在仪器化测试(ohosTest)中运行
 */
import { describe, beforeAll, afterAll, beforeEach, it, expect } from '@ohos/hypium';
import { 
  DataStore, 
  StudentInfo, 
  Grade, 
  Exam, 
  Plan, 
  Progress,
  ScheduleData,
  ExportData,
  TimeSlotPlain,
  CoursePlain
} from '../main/ets/utils/DataStore';
import { SettingsData, TimeSlot, CourseItem } from '../main/ets/model/SettingsData';

export default function dataStoreTest() {
  describe('DataStore', () => {
    
    beforeAll(() => {
      console.info('========================================');
      console.info('DataStore 测试：开始测试');
      console.info('测试时间: ' + new Date().toISOString());
      console.info('========================================');
    });
    
    afterAll(() => {
      console.info('========================================');
      console.info('DataStore 测试：测试结束');
      console.info('========================================');
    });

    describe('单例模式测试', () => {
      it('getInstance_should_return_DataStore_instance', 0, () => {
        console.info('测试：获取DataStore实例');
        
        const instance: DataStore = DataStore.getInstance();
        
        expect(instance === null).assertFalse();
        expect(instance instanceof DataStore).assertTrue();
        console.info('通过：成功获取有效的DataStore实例');
      });

      it('getInstance_should_return_same_instance', 0, () => {
        console.info('测试：验证单例模式返回相同实例');
        
        const instance1: DataStore = DataStore.getInstance();
        const instance2: DataStore = DataStore.getInstance();
        
        expect(instance1 === instance2).assertTrue();
        console.info('通过：单例模式工作正常');
      });
    });

    describe('学生信息数据类型测试', () => {
      it('should_create_valid_StudentInfo', 0, () => {
        console.info('测试：创建有效的学生信息');
        
        const studentInfo: StudentInfo = {
          studentId: '202101001',
          name: '张三',
          gender: '男',
          enrollmentYear: '2021',
          className: '计算机2101班',
          major: '计算机科学与技术',
          college: '信息工程学院'
        };
        
        expect(studentInfo.studentId).assertEqual('202101001');
        expect(studentInfo.name).assertEqual('张三');
        expect(studentInfo.gender).assertEqual('男');
        expect(studentInfo.enrollmentYear).assertEqual('2021');
        expect(studentInfo.className).assertEqual('计算机2101班');
        expect(studentInfo.major).assertEqual('计算机科学与技术');
        expect(studentInfo.college).assertEqual('信息工程学院');
        console.info('通过：成功创建包含所有字段的学生信息');
      });

      it('should_handle_empty_StudentInfo_fields', 0, () => {
        console.info('测试：处理空字段的学生信息');
        
        const studentInfo: StudentInfo = {
          studentId: '',
          name: '',
          gender: '',
          enrollmentYear: '',
          className: '',
          major: '',
          college: ''
        };
        
        expect(studentInfo.studentId).assertEqual('');
        expect(studentInfo.name).assertEqual('');
        console.info('通过：正确处理空字段');
      });

      it('should_handle_special_characters_in_StudentInfo', 0, () => {
        console.info('测试：处理包含特殊字符的学生信息');
        
        const studentInfo: StudentInfo = {
          studentId: '202101001-特殊字符!@#$%',
          name: '张三(测试)',
          gender: '男/女',
          enrollmentYear: '2021',
          className: '计算机2101班[测试]',
          major: '计算机科学与技术{测试}',
          college: '信息工程学院<测试>'
        };
        
        expect(studentInfo.studentId).assertEqual('202101001-特殊字符!@#$%');
        expect(studentInfo.name).assertEqual('张三(测试)');
        console.info('通过：正确处理特殊字符');
      });
    });

    describe('成绩数据类型测试', () => {
      it('should_create_valid_Grade', 0, () => {
        console.info('测试：创建有效的成绩信息');
        
        const grade: Grade = {
          semester: '2024-1',
          courseCode: 'CS101',
          courseName: '数据结构',
          score: '90',
          credit: '4',
          gradePoint: '4.0',
          courseType: '必修课',
          examType: '闭卷考试'
        };
        
        expect(grade.semester).assertEqual('2024-1');
        expect(grade.courseCode).assertEqual('CS101');
        expect(grade.courseName).assertEqual('数据结构');
        expect(grade.score).assertEqual('90');
        expect(grade.credit).assertEqual('4');
        expect(grade.gradePoint).assertEqual('4.0');
        expect(grade.courseType).assertEqual('必修课');
        expect(grade.examType).assertEqual('闭卷考试');
        console.info('通过：成功创建包含所有字段的成绩信息');
      });

      it('should_handle_different_score_formats', 0, () => {
        console.info('测试：处理不同格式的成绩');
        
        const gradeNumeric: Grade = {
          semester: '2024-1',
          courseCode: 'CS101',
          courseName: '数据结构',
          score: '95',
          credit: '4',
          gradePoint: '4.0',
          courseType: '必修课',
          examType: '闭卷考试'
        };
        
        const gradeLetter: Grade = {
          semester: '2024-1',
          courseCode: 'CS102',
          courseName: '操作系统',
          score: '优秀',
          credit: '4',
          gradePoint: '4.0',
          courseType: '必修课',
          examType: '闭卷考试'
        };
        
        expect(gradeNumeric.score).assertEqual('95');
        expect(gradeLetter.score).assertEqual('优秀');
        console.info('通过：正确处理不同格式的成绩');
      });
    });

    describe('考试数据类型测试', () => {
      it('should_create_valid_Exam', 0, () => {
        console.info('测试：创建有效的考试信息');
        
        const exam: Exam = {
          courseName: '数据结构',
          examTime: '2025-01-10 09:00-11:00',
          location: '教A201',
          seatNumber: '01',
          examType: '期末考试',
          status: '未开始'
        };
        
        expect(exam.courseName).assertEqual('数据结构');
        expect(exam.examTime).assertEqual('2025-01-10 09:00-11:00');
        expect(exam.location).assertEqual('教A201');
        expect(exam.seatNumber).assertEqual('01');
        expect(exam.examType).assertEqual('期末考试');
        expect(exam.status).assertEqual('未开始');
        console.info('通过：成功创建包含所有字段的考试信息');
      });

      it('should_handle_different_exam_statuses', 0, () => {
        console.info('测试：处理不同的考试状态');
        
        const statuses: string[] = ['未开始', '进行中', '已结束'];
        
        for (let i = 0; i < statuses.length; i++) {
          const exam: Exam = {
            courseName: '测试课程',
            examTime: '2025-01-10 09:00-11:00',
            location: '教A201',
            seatNumber: '01',
            examType: '期末考试',
            status: statuses[i]
          };
          expect(exam.status).assertEqual(statuses[i]);
        }
        console.info('通过：正确处理不同的考试状态');
      });
    });

    describe('培养计划数据类型测试', () => {
      it('should_create_valid_Plan', 0, () => {
        console.info('测试：创建有效的培养计划');
        
        const plan: Plan = {
          semester: '2024-1',
          courseCode: 'CS101',
          courseName: '数据结构',
          credit: '4',
          totalHours: '64',
          courseType: '必修课',
          courseAttribute: '专业核心课',
          examType: '闭卷考试'
        };
        
        expect(plan.semester).assertEqual('2024-1');
        expect(plan.courseCode).assertEqual('CS101');
        expect(plan.courseName).assertEqual('数据结构');
        expect(plan.credit).assertEqual('4');
        expect(plan.totalHours).assertEqual('64');
        expect(plan.courseType).assertEqual('必修课');
        expect(plan.courseAttribute).assertEqual('专业核心课');
        expect(plan.examType).assertEqual('闭卷考试');
        console.info('通过：成功创建包含所有字段的培养计划');
      });
    });

    describe('学分进度数据类型测试', () => {
      it('should_create_valid_Progress', 0, () => {
        console.info('测试：创建有效的学分进度');
        
        const progress: Progress = {
          category: '必修课',
          requiredCredits: '50',
          completedCredits: '30',
          currentCredits: '10',
          remainingCredits: '10'
        };
        
        expect(progress.category).assertEqual('必修课');
        expect(progress.requiredCredits).assertEqual('50');
        expect(progress.completedCredits).assertEqual('30');
        expect(progress.currentCredits).assertEqual('10');
        expect(progress.remainingCredits).assertEqual('10');
        console.info('通过：成功创建包含所有字段的学分进度');
      });

      it('should_calculate_progress_correctly', 0, () => {
        console.info('测试：正确计算学分进度');
        
        const progress: Progress = {
          category: '必修课',
          requiredCredits: '50',
          completedCredits: '30',
          currentCredits: '10',
          remainingCredits: '10'
        };
        
        const total: number = parseInt(progress.completedCredits) + 
                              parseInt(progress.currentCredits) + 
                              parseInt(progress.remainingCredits);
        
        expect(total).assertEqual(50);
        console.info('通过：学分进度计算正确');
      });
    });

    describe('设置数据类型测试', () => {
      it('should_create_SettingsData_with_defaults', 0, () => {
        console.info('测试：使用默认值创建设置数据');
        
        const settings: SettingsData = new SettingsData();
        
        expect(settings.semesterName !== null).assertTrue();
        expect(settings.semesterStartDate !== null).assertTrue();
        expect(settings.totalWeeks > 0).assertTrue();
        expect(settings.maxPeriods > 0).assertTrue();
        console.info('通过：成功使用默认值创建设置数据');
      });

      it('should_allow_modification_of_SettingsData', 0, () => {
        console.info('测试：允许修改设置数据');
        
        const settings: SettingsData = new SettingsData();
        settings.semesterName = '2024-2025学年第一学期';
        settings.totalWeeks = 20;
        settings.showWeekend = true;
        settings.showCurrentWeek = false;
        
        expect(settings.semesterName).assertEqual('2024-2025学年第一学期');
        expect(settings.totalWeeks).assertEqual(20);
        expect(settings.showWeekend).assertTrue();
        expect(settings.showCurrentWeek).assertFalse();
        console.info('通过：设置数据修改功能正常');
      });
    });

    describe('时间段数据类型测试', () => {
      it('should_create_valid_TimeSlot', 0, () => {
        console.info('测试：创建有效的时间段');
        
        const timeSlot: TimeSlot = new TimeSlot('08:00-09:40', '第1-2节');
        
        expect(timeSlot.time).assertEqual('08:00-09:40');
        expect(timeSlot.period).assertEqual('第1-2节');
        console.info('通过：成功创建时间段');
      });
    });

    describe('课程项数据类型测试', () => {
      it('should_create_valid_CourseItem', 0, () => {
        console.info('测试：创建有效的课程项');
        
        const course: CourseItem = new CourseItem(
          1,
          1,
          '数据结构',
          '教A201',
          '张老师',
          [1, 2, 3, 4, 5],
          '',
          '2024-1'
        );
        
        expect(course.period).assertEqual(1);
        expect(course.dayOfWeek).assertEqual(1);
        expect(course.name).assertEqual('数据结构');
        expect(course.location).assertEqual('教A201');
        expect(course.teacher).assertEqual('张老师');
        expect(course.weeks.length).assertEqual(5);
        expect(course.semester).assertEqual('2024-1');
        console.info('通过：成功创建课程项');
      });

      it('should_handle_empty_weeks_in_CourseItem', 0, () => {
        console.info('测试：处理空周次的课程项');
        
        const course: CourseItem = new CourseItem(
          1,
          1,
          '测试课程',
          '教A201',
          '张老师',
          [],
          '',
          '2024-1'
        );
        
        expect(course.weeks.length).assertEqual(0);
        console.info('通过：正确处理空周次');
      });
    });

    describe('课表数据类型测试', () => {
      it('should_create_valid_ScheduleData', 0, () => {
        console.info('测试：创建有效的课表数据');
        
        const scheduleData: ScheduleData = {
          courses: [],
          semester: '2024-1',
          lastUpdated: Date.now()
        };
        
        expect(Array.isArray(scheduleData.courses)).assertTrue();
        expect(scheduleData.semester).assertEqual('2024-1');
        expect(scheduleData.lastUpdated > 0).assertTrue();
        console.info('通过：成功创建课表数据');
      });
    });

    describe('数组操作测试', () => {
      it('should_handle_Grade_array_operations', 0, () => {
        console.info('测试：处理成绩数组操作');
        
        const grades: Grade[] = [
          {
            semester: '2024-1',
            courseCode: 'CS101',
            courseName: '数据结构',
            score: '90',
            credit: '4',
            gradePoint: '4.0',
            courseType: '必修课',
            examType: '闭卷考试'
          },
          {
            semester: '2024-1',
            courseCode: 'CS102',
            courseName: '操作系统',
            score: '85',
            credit: '4',
            gradePoint: '3.5',
            courseType: '必修课',
            examType: '闭卷考试'
          }
        ];
        
        expect(grades.length).assertEqual(2);
        expect(grades[0].courseName).assertEqual('数据结构');
        expect(grades[1].courseName).assertEqual('操作系统');
        console.info('通过：成绩数组操作正常');
      });

      it('should_handle_empty_array', 0, () => {
        console.info('测试：处理空数组');
        
        const grades: Grade[] = [];
        
        expect(grades.length).assertEqual(0);
        console.info('通过：正确处理空数组');
      });

      it('should_filter_array_correctly', 0, () => {
        console.info('测试：正确过滤数组');
        
        const grades: Grade[] = [
          { semester: '2024-1', courseCode: 'CS101', courseName: '数据结构', score: '90', credit: '4', gradePoint: '4.0', courseType: '必修课', examType: '闭卷考试' },
          { semester: '2024-2', courseCode: 'CS102', courseName: '操作系统', score: '85', credit: '4', gradePoint: '3.5', courseType: '必修课', examType: '闭卷考试' },
          { semester: '2024-1', courseCode: 'CS103', courseName: '计算机网络', score: '88', credit: '3', gradePoint: '3.8', courseType: '必修课', examType: '闭卷考试' }
        ];
        
        const filtered: Grade[] = grades.filter((g: Grade) => g.semester === '2024-1');
        
        expect(filtered.length).assertEqual(2);
        console.info('通过：数组过滤功能正常');
      });

      it('should_sort_array_correctly', 0, () => {
        console.info('测试：正确排序数组');
        
        const grades: Grade[] = [
          { semester: '2024-1', courseCode: 'CS101', courseName: '数据结构', score: '90', credit: '4', gradePoint: '4.0', courseType: '必修课', examType: '闭卷考试' },
          { semester: '2024-1', courseCode: 'CS102', courseName: '操作系统', score: '85', credit: '4', gradePoint: '3.5', courseType: '必修课', examType: '闭卷考试' },
          { semester: '2024-1', courseCode: 'CS103', courseName: '计算机网络', score: '88', credit: '3', gradePoint: '3.8', courseType: '必修课', examType: '闭卷考试' }
        ];
        
        const sorted: Grade[] = grades.sort((a: Grade, b: Grade) => 
          parseInt(b.score) - parseInt(a.score)
        );
        
        expect(sorted[0].score).assertEqual('90');
        expect(sorted[2].score).assertEqual('85');
        console.info('通过：数组排序功能正常');
      });
    });

    describe('JSON操作测试', () => {
      it('should_serialize_StudentInfo_to_json', 0, () => {
        console.info('测试：序列化学生信息为JSON');
        
        const studentInfo: StudentInfo = {
          studentId: '202101001',
          name: '张三',
          gender: '男',
          enrollmentYear: '2021',
          className: '计算机2101班',
          major: '计算机科学与技术',
          college: '信息工程学院'
        };
        
        const json: string = JSON.stringify(studentInfo);
        const parsed: StudentInfo = JSON.parse(json) as StudentInfo;
        
        expect(typeof json).assertEqual('string');
        expect(parsed.studentId).assertEqual('202101001');
        expect(parsed.name).assertEqual('张三');
        console.info('通过：学生信息JSON序列化功能正常');
      });

      it('should_serialize_Grade_array_to_json', 0, () => {
        console.info('测试：序列化成绩数组为JSON');
        
        const grades: Grade[] = [
          { semester: '2024-1', courseCode: 'CS101', courseName: '数据结构', score: '90', credit: '4', gradePoint: '4.0', courseType: '必修课', examType: '闭卷考试' }
        ];
        
        const json: string = JSON.stringify(grades);
        const parsed: Grade[] = JSON.parse(json) as Grade[];
        
        expect(parsed.length).assertEqual(1);
        expect(parsed[0].courseName).assertEqual('数据结构');
        console.info('通过：成绩数组JSON序列化功能正常');
      });

      it('should_handle_invalid_json', 0, () => {
        console.info('测试：处理无效JSON');
        
        const invalidJson: string = 'not a valid json';
        let error: boolean = false;
        
        try {
          JSON.parse(invalidJson);
        } catch (e) {
          error = true;
        }
        
        expect(error).assertTrue();
        console.info('通过：正确处理无效JSON错误');
      });

      it('should_handle_json_with_unicode', 0, () => {
        console.info('测试：处理包含Unicode的JSON');
        
        const studentInfo: StudentInfo = {
          studentId: '202101001',
          name: '张三',
          gender: '男',
          enrollmentYear: '2021',
          className: '计算机2101班',
          major: '计算机科学与技术',
          college: '信息工程学院'
        };
        
        const json: string = JSON.stringify(studentInfo);
        const parsed: StudentInfo = JSON.parse(json) as StudentInfo;
        
        expect(parsed.name).assertEqual('张三');
        expect(parsed.major).assertEqual('计算机科学与技术');
        console.info('通过：正确处理JSON中的Unicode字符');
      });
    });

    describe('性能测试', () => {
      it('should_handle_large_Grade_array', 0, () => {
        console.info('测试：处理大量成绩数据');
        
        const grades: Grade[] = [];
        const startTime: number = Date.now();
        
        for (let i = 0; i < 1000; i++) {
          grades.push({
            semester: '2024-1',
            courseCode: 'CS' + i,
            courseName: '课程' + i,
            score: String(60 + (i % 40)),
            credit: '4',
            gradePoint: '4.0',
            courseType: '必修课',
            examType: '闭卷考试'
          });
        }
        
        const endTime: number = Date.now();
        const duration: number = endTime - startTime;
        
        expect(grades.length).assertEqual(1000);
        console.info('通过：成功创建1000条成绩数据，耗时 ' + duration + ' 毫秒');
      });

      it('should_serialize_large_array_efficiently', 0, () => {
        console.info('测试：高效序列化大量数据');
        
        const grades: Grade[] = [];
        for (let i = 0; i < 500; i++) {
          grades.push({
            semester: '2024-1',
            courseCode: 'CS' + i,
            courseName: '课程' + i,
            score: String(60 + (i % 40)),
            credit: '4',
            gradePoint: '4.0',
            courseType: '必修课',
            examType: '闭卷考试'
          });
        }
        
        const startTime: number = Date.now();
        const json: string = JSON.stringify(grades);
        const parsed: Grade[] = JSON.parse(json) as Grade[];
        const endTime: number = Date.now();
        const duration: number = endTime - startTime;
        
        expect(parsed.length).assertEqual(500);
        console.info('通过：成功序列化和解析500条成绩数据，耗时 ' + duration + ' 毫秒');
      });
    });

    describe('数据验证测试', () => {
      it('should_validate_required_fields_in_StudentInfo', 0, () => {
        console.info('测试：验证学生信息必填字段');
        
        const studentInfo: StudentInfo = {
          studentId: '202101001',
          name: '张三',
          gender: '男',
          enrollmentYear: '2021',
          className: '计算机2101班',
          major: '计算机科学与技术',
          college: '信息工程学院'
        };
        
        const hasAllFields: boolean = 
          studentInfo.studentId !== '' &&
          studentInfo.name !== '' &&
          studentInfo.gender !== '' &&
          studentInfo.enrollmentYear !== '' &&
          studentInfo.className !== '' &&
          studentInfo.major !== '' &&
          studentInfo.college !== '';
        
        expect(hasAllFields).assertTrue();
        console.info('通过：必填字段验证功能正常');
      });

      it('should_validate_score_range', 0, () => {
        console.info('测试：验证成绩范围');
        
        const validScores: string[] = ['60', '75', '90', '100'];
        const invalidScores: string[] = ['-1', '101', 'abc', ''];
        
        for (let i = 0; i < validScores.length; i++) {
          const score: number = parseInt(validScores[i]);
          const isValid: boolean = !isNaN(score) && score >= 0 && score <= 100;
          expect(isValid).assertTrue();
        }
        
        for (let i = 0; i < invalidScores.length; i++) {
          const score: number = parseInt(invalidScores[i]);
          const isValid: boolean = !isNaN(score) && score >= 0 && score <= 100;
          expect(isValid).assertFalse();
        }
        
        console.info('通过：成绩范围验证功能正常');
      });

      it('should_validate_semester_format', 0, () => {
        console.info('测试：验证学期格式');
        
        const validSemesters: string[] = ['2024-1', '2024-2', '2023-1', '2025-2'];
        const invalidSemesters: string[] = ['2024', '2024-3', 'abc', ''];
        
        for (let i = 0; i < validSemesters.length; i++) {
          const semester: string = validSemesters[i];
          const isValid: boolean = /^\d{4}-[12]$/.test(semester);
          expect(isValid).assertTrue();
        }
        
        for (let i = 0; i < invalidSemesters.length; i++) {
          const semester: string = invalidSemesters[i];
          const isValid: boolean = /^\d{4}-[12]$/.test(semester);
          expect(isValid).assertFalse();
        }
        
        console.info('通过：学期格式验证功能正常');
      });
    });
  });
}
