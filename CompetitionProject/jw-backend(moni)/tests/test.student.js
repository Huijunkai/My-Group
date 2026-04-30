/**
 * Student API 单元测试
 * 测试模块：学生数据模块
 * 测试内容：学生信息、课表、成绩、考试安排、培养计划、学习进度
 */
const { getStudentInfo, getTimetable, getGrades, getExamSchedule, getSemesterPlan, getStudyProgress } = require('../src/api/student');
const assert = require('assert');

describe('Student API - 学生数据模块', () => {
    // 测试固件
    const mockCookies = [
        'JSESSIONID=MOCK_test_1234567890; Path=/',
        'studentId=202101001; Path=/'
    ];

    describe('getStudentInfo() - 功能正确性测试', () => {
        it('【正常输入】应该返回学生基本信息', async () => {
            // Arrange
            const cookies = mockCookies;

            // Act
            const info = await getStudentInfo(cookies);

            // Assert
            assert.ok(info, '学生信息不应该为空');
            assert.strictEqual(info.studentId, '202101001', '学号应该匹配');
            assert.ok(info.name, '姓名不应该为空');
            assert.ok(info.gender, '性别不应该为空');
            assert.ok(info.enrollmentYear, '入学年份不应该为空');
            assert.ok(info.className, '班级不应该为空');
            assert.ok(info.major, '专业不应该为空');
            assert.ok(info.college, '学院不应该为空');
        });

        it('【返回值验证】应该返回正确的数据类型', async () => {
            // Arrange
            const cookies = mockCookies;

            // Act
            const info = await getStudentInfo(cookies);

            // Assert
            assert.strictEqual(typeof info.studentId, 'string', '学号应该是字符串');
            assert.strictEqual(typeof info.name, 'string', '姓名应该是字符串');
            assert.strictEqual(typeof info.gender, 'string', '性别应该是字符串');
            assert.strictEqual(typeof info.enrollmentYear, 'string', '入学年份应该是字符串');
            assert.strictEqual(typeof info.className, 'string', '班级应该是字符串');
            assert.strictEqual(typeof info.major, 'string', '专业应该是字符串');
            assert.strictEqual(typeof info.college, 'string', '学院应该是字符串');
        });

        it('【数据一致性】性别值应该是有效的', async () => {
            // Arrange
            const cookies = mockCookies;

            // Act
            const info = await getStudentInfo(cookies);

            // Assert
            assert.ok(['男', '女'].includes(info.gender), `性别 "${info.gender}" 应该是男或女`);
        });
    });

    describe('getTimetable() - 功能正确性测试', () => {
        it('【正常输入】应该返回课程表数组', async () => {
            // Arrange
            const cookies = mockCookies;

            // Act
            const timetable = await getTimetable(cookies);

            // Assert
            assert.ok(Array.isArray(timetable), '课程表应该是数组');
            assert.ok(timetable.length > 0, '课程表不应为空');
        });

        it('【返回值验证】每条课程记录应包含必要字段', async () => {
            // Arrange
            const cookies = mockCookies;

            // Act
            const timetable = await getTimetable(cookies);
            const course = timetable[0];

            // Assert
            assert.ok(course.name, '课程名称不应该为空');
            assert.ok(course.dayOfWeek, '星期几不应该为空');
            assert.ok(course.period, '节次不应该为空');
            assert.ok(course.teacher, '教师不应该为空');
            assert.ok(course.location, '地点不应该为空');
            assert.ok(course.courseType, '课程类型不应该为空');
        });

        it('【数据筛选】应该支持按学期筛选', async () => {
            // Arrange
            const cookies = mockCookies;
            const semester = '2025-1';

            // Act
            const timetable = await getTimetable(cookies, semester);

            // Assert
            assert.ok(Array.isArray(timetable), '筛选结果应该是数组');
        });

        it('【数据一致性】星期几的值应该在有效范围内', async () => {
            // Arrange
            const cookies = mockCookies;
            const validDays = ['星期一', '星期二', '星期三', '星期四', '星期五', '星期六', '星期日'];

            // Act
            const timetable = await getTimetable(cookies);

            // Assert
            timetable.forEach(course => {
                assert.ok(validDays.includes(course.dayOfWeek), `${course.dayOfWeek} 不是有效的星期几`);
            });
        });

        it('【数据一致性】节次格式应该正确', async () => {
            // Arrange
            const cookies = mockCookies;
            const validPeriods = ['01-02', '03-04', '05-06', '07-08', '09-10', '11-12'];

            // Act
            const timetable = await getTimetable(cookies);

            // Assert
            timetable.forEach(course => {
                assert.ok(validPeriods.includes(course.period), `${course.period} 不是有效的节次格式`);
            });
        });
    });

    describe('getGrades() - 功能正确性测试', () => {
        it('【正常输入】应该返回成绩对象（按学期分组）', async () => {
            // Arrange
            const cookies = mockCookies;

            // Act
            const grades = await getGrades(cookies);

            // Assert
            assert.ok(grades, '成绩数据不应该为空');
            assert.strictEqual(typeof grades, 'object', '成绩应该是对象');
            const semesterKeys = Object.keys(grades);
            assert.ok(semesterKeys.length > 0, '至少有一个学期的成绩');
        });

        it('【返回值验证】每个学期应该包含成绩数组', async () => {
            // Arrange
            const cookies = mockCookies;

            // Act
            const grades = await getGrades(cookies);

            // Assert
            Object.values(grades).forEach(semesterGrades => {
                assert.ok(Array.isArray(semesterGrades), '学期成绩应该是数组');
            });
        });

        it('【返回值验证】成绩应该包含必要字段', async () => {
            // Arrange
            const cookies = mockCookies;

            // Act
            const grades = await getGrades(cookies);
            const firstSemester = Object.values(grades)[0];

            // Assert
            if (firstSemester && firstSemester.length > 0) {
                const grade = firstSemester[0];
                assert.ok(grade.courseName || grade.name, '课程名称不应该为空');
                assert.ok(grade.score !== undefined, '分数不应该为空');
                assert.ok(grade.credit !== undefined, '学分不应该为空');
            }
        });

        it('【数据筛选】应该支持按学期筛选成绩', async () => {
            // Arrange
            const cookies = mockCookies;
            const semester = '2024-1';

            // Act
            const grades = await getGrades(cookies, semester);

            // Assert
            if (grades && Object.keys(grades).length > 0) {
                assert.ok(grades[semester] !== undefined || Object.keys(grades).length === 1, '应该返回指定学期的成绩');
            }
        });
    });

    describe('getExamSchedule() - 功能正确性测试', () => {
        it('【正常输入】应该返回考试安排数组', async () => {
            // Arrange
            const cookies = mockCookies;

            // Act
            const exams = await getExamSchedule(cookies);

            // Assert
            assert.ok(Array.isArray(exams), '考试安排应该是数组');
        });

        it('【返回值验证】每条考试记录应包含必要字段', async () => {
            // Arrange
            const cookies = mockCookies;

            // Act
            const exams = await getExamSchedule(cookies);

            // Assert
            if (exams && exams.length > 0) {
                const exam = exams[0];
                assert.ok(exam.courseName || exam.name, '课程名称不应该为空');
                assert.ok(exam.examTime || exam.time, '考试时间不应该为空');
                assert.ok(exam.location || exam.place, '考试地点不应该为空');
            }
        });
    });

    describe('getSemesterPlan() - 功能正确性测试', () => {
        it('【正常输入】应该返回培养计划对象', async () => {
            // Arrange
            const cookies = mockCookies;

            // Act
            const plans = await getSemesterPlan(cookies);

            // Assert
            assert.ok(plans, '培养计划不应该为空');
            assert.strictEqual(typeof plans, 'object', '培养计划应该是对象');
        });

        it('【返回值验证】每个学期计划应包含课程信息', async () => {
            // Arrange
            const cookies = mockCookies;

            // Act
            const plans = await getSemesterPlan(cookies);
            const firstSemester = Object.values(plans)[0];

            // Assert
            if (firstSemester && firstSemester.length > 0) {
                const plan = firstSemester[0];
                assert.ok(plan.courseName || plan.name, '课程名称不应该为空');
                assert.ok(plan.credit, '学分不应该为空');
            }
        });

        it('【数据一致性】学分应该是有效的数值', async () => {
            // Arrange
            const cookies = mockCookies;

            // Act
            const plans = await getSemesterPlan(cookies);

            // Assert
            Object.values(plans).forEach(semesterPlans => {
                semesterPlans.forEach(plan => {
                    if (plan.credit !== undefined) {
                        const credit = parseFloat(plan.credit);
                        assert.ok(!isNaN(credit), `学分 "${plan.credit}" 应该是有效数值`);
                        assert.ok(credit > 0, `学分 ${credit} 应该大于0`);
                    }
                });
            });
        });
    });

    describe('getStudyProgress() - 功能正确性测试', () => {
        it('【正常输入】应该返回学习进度数组', async () => {
            // Arrange
            const cookies = mockCookies;

            // Act
            const progress = await getStudyProgress(cookies);

            // Assert
            assert.ok(Array.isArray(progress), '进度应该是数组');
        });

        it('【返回值验证】每条进度记录应包含学分信息', async () => {
            // Arrange
            const cookies = mockCookies;

            // Act
            const progress = await getStudyProgress(cookies);

            // Assert
            if (progress && progress.length > 0) {
                const item = progress[0];
                assert.ok(item.category, '类别不应该为空');
                assert.ok(item.requiredCredits !== undefined, '要求学分不应该为空');
                assert.ok(item.completedCredits !== undefined, '已完成学分不应该为空');
            }
        });

        it('【数据一致性】已完成学分应该是有效的数值', async () => {
            // Arrange
            const cookies = mockCookies;

            // Act
            const progress = await getStudyProgress(cookies);

            // Assert
            progress.forEach(item => {
                if (item.requiredCredits !== undefined && item.completedCredits !== undefined) {
                    const completed = parseFloat(item.completedCredits);
                    const required = parseFloat(item.requiredCredits);
                    assert.ok(!isNaN(completed), `类别 "${item.category}" 已完成学分应该是有效数值`);
                    assert.ok(!isNaN(required), `类别 "${item.category}" 要求学分应该是有效数值`);
                    assert.ok(completed >= 0, `类别 "${item.category}" 已完成学分应该大于等于0`);
                    assert.ok(required >= 0, `类别 "${item.category}" 要求学分应该大于等于0`);
                }
            });
        });
    });

    describe('性能测试', () => {
        it('【性能】getStudentInfo 应该在合理时间内返回', async () => {
            // Arrange
            const cookies = mockCookies;
            const maxResponseTime = 1000;

            // Act
            const startTime = Date.now();
            await getStudentInfo(cookies);
            const elapsedTime = Date.now() - startTime;

            // Assert
            assert.ok(elapsedTime < maxResponseTime, `响应时间 ${elapsedTime}ms 应该小于 ${maxResponseTime}ms`);
        });

        it('【性能】getTimetable 应该在合理时间内返回', async () => {
            // Arrange
            const cookies = mockCookies;
            const maxResponseTime = 1000;

            // Act
            const startTime = Date.now();
            await getTimetable(cookies);
            const elapsedTime = Date.now() - startTime;

            // Assert
            assert.ok(elapsedTime < maxResponseTime, `响应时间 ${elapsedTime}ms 应该小于 ${maxResponseTime}ms`);
        });
    });
});