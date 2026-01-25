const express = require('express');
const cors = require('cors');
const { login } = require('./src/api/auth');
const { getStudentInfo, getTimetable, getGrades, getExamSchedule } = require('./src/api/student');
const { initDatabase } = require('./src/db');
const { syncStudent, syncCourses, syncGrades, syncExams } = require('./src/db/sync');
const { Student, Course, Grade, Exam } = require('./src/db/models');

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
            Promise.all([
                getTimetable(cookies).then(data => data && syncCourses(username, data)),
                getGrades(cookies).then(data => data && syncGrades(username, data)),
                getExamSchedule(cookies).then(data => data && syncExams(username, data))
            ]).catch(err => console.error('Background sync error:', err));

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
        res.status(500).json({ success: false, message: '服务器内部错误' });
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
        const [courses, grades, exams] = await Promise.all([
            Course.findAll({ where: { studentId } }),
            Grade.findAll({ where: { studentId } }),
            Exam.findAll({ where: { studentId } })
        ]);

        res.json({
            success: true,
            data: {
                info: student,
                courses,
                grades,
                exams
            }
        });
    } catch (error) {
        res.status(500).json({ success: false, message: error.message });
    }
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server is running on port ${PORT}`);
});
