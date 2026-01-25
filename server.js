const express = require('express');
const cors = require('cors');
const { login } = require('./src/api/auth');
const { getStudentInfo, getTimetable, getGrades, getExamSchedule, getSemesterPlan, getStudyProgress } = require('./src/api/student');
const { syncStudent, syncCourses, syncGrades, syncExams, syncPlans, syncProgress } = require('./src/db/sync');
const { Student, Course, Grade, Exam, Plan, Progress } = require('./src/db/models');

const { initDatabase } = require('./src/db');

const app = express();
app.use(cors());
app.use(express.json());

// 端口配置，Railway 会自动注入 PORT 环境变量
const PORT = process.env.PORT || 3000;

// 初始化数据库
initDatabase().then(() => {
    console.log('Database initialized');
});

/**
 * 根路由测试
 */
app.get('/', (req, res) => {
    res.json({ message: '教务系统同步服务已启动', status: 'running' });
});

/**
 * 同步接口：登录并同步所有数据到数据库
 * POST /api/sync
 * Body: { username, password }
 */
app.post('/api/sync', async (req, res) => {
    const { username, password } = req.body;

    if (!username || !password) {
        return res.status(400).json({ success: false, message: '请提供学号和密码' });
    }

    try {
        console.log(`正在同步学生数据: ${username}`);
        const loginResult = await login(username, password);

        if (!loginResult.success) {
            return res.status(401).json({ success: false, message: loginResult.message });
        }

        const cookies = loginResult.cookies;

        // 异步执行同步逻辑，不阻塞响应（或者等待完成，取决于需求）
        // 这里我们选择等待基本信息完成，其他后台同步
        const info = await getStudentInfo(cookies);
        if (info) {
            await syncStudent(username, info);
            
            // 后台静默同步其他数据
            // 注意：这里不使用 await，让它在后台运行
            // 如果需要确保所有数据都同步完再返回，可以在这里加 await
            // 但考虑到爬虫速度，建议先返回基本信息，其他让前端轮询或者下次进入时获取
            
            // 为了防止 Promise.all 抛出未捕获异常导致进程崩溃，这里单独处理每个 Promise
            const syncTasks = [
                getTimetable(cookies).then(data => data && syncCourses(username, data)).catch(e => console.error('Sync courses failed:', e)),
                getGrades(cookies).then(data => data && syncGrades(username, data)).catch(e => console.error('Sync grades failed:', e)),
                getExamSchedule(cookies).then(data => data && syncExams(username, data)).catch(e => console.error('Sync exams failed:', e)),
                getSemesterPlan(cookies).then(data => data && syncPlans(username, data)).catch(e => console.error('Sync plans failed:', e)),
                getStudyProgress(cookies).then(data => data && syncProgress(username, data)).catch(e => console.error('Sync progress failed:', e))
            ];
            
            // 触发任务但不等待
            Promise.all(syncTasks).then(() => console.log(`后台同步完成: ${username}`));

            return res.json({
                success: true,
                message: '登录成功，数据正在后台同步中',
                student: info
            });
        } else {
            return res.status(500).json({ success: false, message: '获取学生信息失败' });
        }
    } catch (error) {
        console.error('Sync error:', error);
        // 确保返回具体的错误信息以便调试
        res.status(500).json({ success: false, message: '服务器内部错误: ' + error.message });
    }
});

/**
 * 查询接口：从数据库获取已缓存的学生信息
 * GET /api/student/:id
 */
app.get('/api/student/:id', async (req, res) => {
    const studentId = req.params.id;
    try {
        const student = await Student.findByPk(studentId, {
            include: [
                // 如果设置了关联可以 include，目前模型是独立的，我们手动查询
            ]
        });

        if (!student) {
            return res.status(404).json({ success: false, message: '未找到该学生缓存数据' });
        }

        // 查询关联数据
        const [courses, grades, exams, plans, progress] = await Promise.all([
            // 课表：按学期 -> 周次 -> 星期 -> 节次排序
            Course.findAll({
                where: { studentId },
                order: [
                    ['semester', 'ASC'],
                    ['week', 'ASC'],
                    ['dayOfWeek', 'ASC'],
                    ['period', 'ASC'],
                    ['name', 'ASC']
                ]
            }),
            // 成绩：按学期 -> 课程编号排序
            Grade.findAll({
                where: { studentId },
                order: [
                    ['semester', 'ASC'],
                    ['courseCode', 'ASC']
                ]
            }),
            // 考试：按时间 -> 课程名排序
            Exam.findAll({
                where: { studentId },
                order: [
                    ['examTime', 'ASC'],
                    ['courseName', 'ASC']
                ]
            }),
            // 学期计划：按学期 -> 课程编号排序
            Plan.findAll({
                where: { studentId },
                order: [
                    ['semester', 'ASC'],
                    ['courseCode', 'ASC']
                ]
            }),
            // 学习完成情况：按分类名排序
            Progress.findAll({
                where: { studentId },
                order: [['category', 'ASC']]
            })
        ]);

        res.json({
            success: true,
            data: {
                info: student,
                courses,
                grades,
                exams,
                plans,
                progress
            }
        });
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
});

/**
 * 学生列表（按学号升序）
 * GET /api/students
 */
app.get('/api/students', async (_req, res) => {
    try {
        const students = await Student.findAll({
            order: [['studentId', 'ASC']]
        });
        res.json({ success: true, data: students });
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server is running on port ${PORT}`);
});
