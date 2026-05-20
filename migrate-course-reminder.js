require('dotenv').config({ path: __dirname + '/.env' });

const { Sequelize, DataTypes } = require('sequelize');

const DB_NAME = process.env.DB_NAME || 'app_db';
const DB_USER = process.env.DB_USER || 'root';
const DB_PASSWORD = process.env.DB_PASSWORD || '021219Hjk!';
const DB_HOST = process.env.DB_HOST || '127.0.0.1';
const DB_PORT = parseInt(process.env.DB_PORT) || 3306;

const sequelize = new Sequelize(DB_NAME, DB_USER, DB_PASSWORD, {
    host: DB_HOST,
    port: DB_PORT,
    dialect: 'mysql',
    logging: false
});

async function migrate() {
    try {
        console.log('开始数据库迁移...');
        console.log(`连接数据库: ${DB_HOST}:${DB_PORT}/${DB_NAME}`);
        
        await sequelize.authenticate();
        console.log('数据库连接成功');
        
        await sequelize.getQueryInterface().createTable('CourseReminderConfig', {
            id: { type: DataTypes.INTEGER, primaryKey: true, autoIncrement: true },
            studentId: { type: DataTypes.STRING(100), unique: true },
            enabled: { type: DataTypes.BOOLEAN, defaultValue: true },
            semesterStartDate: { type: DataTypes.STRING(50), allowNull: true },
            currentWeek: { type: DataTypes.INTEGER, defaultValue: 1 },
            beforeClassMinutes: { type: DataTypes.INTEGER, defaultValue: 15 },
            remindBeforeClass: { type: DataTypes.BOOLEAN, defaultValue: true },
            remindTomorrowCourse: { type: DataTypes.BOOLEAN, defaultValue: true },
            tomorrowHour: { type: DataTypes.INTEGER, defaultValue: 21 },
            tomorrowMinute: { type: DataTypes.INTEGER, defaultValue: 0 },
            createdAt: { type: DataTypes.DATE, defaultValue: DataTypes.NOW },
            updatedAt: { type: DataTypes.DATE, defaultValue: DataTypes.NOW }
        });
        
        console.log('✓ 成功创建 CourseReminderConfig 表');
        
        try {
            await sequelize.getQueryInterface().addColumn('Student', 'semesterStartDate', {
                type: DataTypes.STRING(50),
                allowNull: true
            });
            console.log('✓ 成功添加 Student.semesterStartDate 字段');
        } catch (e) {
            if (e.message.includes('Duplicate column') || e.message.includes('already exists')) {
                console.log('✓ Student.semesterStartDate 字段已存在，跳过');
            } else {
                console.log('添加 Student.semesterStartDate 字段时出错:', e.message);
            }
        }
        
        console.log('数据库迁移完成！');
        await sequelize.close();
        process.exit(0);
    } catch (error) {
        if (error.message.includes('already exists')) {
            console.log('CourseReminderConfig 表已存在，尝试添加字段...');
            
            try {
                await sequelize.getQueryInterface().addColumn('Student', 'semesterStartDate', {
                    type: DataTypes.STRING(50),
                    allowNull: true
                });
                console.log('✓ 成功添加 Student.semesterStartDate 字段');
            } catch (e) {
                if (e.message.includes('Duplicate column') || e.message.includes('already exists')) {
                    console.log('✓ Student.semesterStartDate 字段已存在，跳过');
                }
            }
            
            console.log('数据库迁移完成！');
            await sequelize.close();
            process.exit(0);
        } else {
            console.error('迁移失败:', error.message);
            await sequelize.close();
            process.exit(1);
        }
    }
}

migrate();
