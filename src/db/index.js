const { Sequelize } = require('sequelize');

const DB_NAME = process.env.DB_NAME || 'app_db';
const DB_USER = process.env.DB_USER || 'root';
const DB_PASSWORD = process.env.DB_PASSWORD || '021219Hjk!';
const DB_HOST = process.env.DB_HOST || '127.0.0.1';
const DB_PORT = parseInt(process.env.DB_PORT) || 3306;

let heartbeatTimer = null;
let isReconnecting = false;
let dbHealthy = false;

const sequelize = new Sequelize(
    DB_NAME,
    DB_USER,
    DB_PASSWORD,
    {
        host: DB_HOST,
        port: DB_PORT,
        dialect: 'mysql',
        logging: false,

        pool: {
            max: 10,
            min: 2,
            acquire: 30000,
            idle: 120000,
            evict: 10000,
            handleDisconnects: true
        },

        define: {
            charset: 'utf8mb4',
            collate: 'utf8mb4_unicode_ci',
            underscored: false,
            freezeTableName: true
        },

        timezone: '+08:00',
        retry: {
            max: 3,
            match: [
                /SequelizeConnectionError/,
                /SequelizeConnectionRefusedError/,
                /SequelizeHostNotReachableError/,
                /SequelizeInvalidConnectionError/,
                /ETIMEDOUT/,
                /ECONNRESET/,
                /ECONNREFUSED/,
                /PROTOCOL_CONNECTION_LOST/
            ]
        }
    }
);

sequelize.addHook('afterDisconnect', (connection) => {
    console.warn(`[${new Date().toLocaleTimeString()}] 数据库连接断开，正在尝试重连...`);
    dbHealthy = false;
    scheduleReconnect();
});

function scheduleReconnect() {
    if (isReconnecting) return;
    isReconnecting = true;

    setTimeout(async () => {
        try {
            await sequelize.authenticate();
            dbHealthy = true;
            isReconnecting = false;
            console.log(`[${new Date().toLocaleTimeString()}] 数据库重连成功`);
        } catch (error) {
            isReconnecting = false;
            console.error(`[${new Date().toLocaleTimeString()}] 数据库重连失败:`, error.message);
            setTimeout(() => scheduleReconnect(), 5000);
        }
    }, 2000);
}

async function checkHealth() {
    try {
        await sequelize.authenticate();
        dbHealthy = true;
        return true;
    } catch (error) {
        dbHealthy = false;
        return false;
    }
}

function startHeartbeat(intervalMs = 30000) {
    if (heartbeatTimer) return;

    heartbeatTimer = setInterval(async () => {
        try {
            await sequelize.query('SELECT 1');
        } catch (error) {
            console.warn(`[${new Date().toLocaleTimeString()}] 数据库心跳检测失败:`, error.message);
            dbHealthy = false;
            if (!isReconnecting) {
                scheduleReconnect();
            }
        }
    }, intervalMs);

    heartbeatTimer.unref();
}

function stopHeartbeat() {
    if (heartbeatTimer) {
        clearInterval(heartbeatTimer);
        heartbeatTimer = null;
    }
}

async function initDatabase() {
    const MAX_INIT_RETRIES = 5;
    const INIT_RETRY_DELAY = 3000;

    for (let attempt = 1; attempt <= MAX_INIT_RETRIES; attempt++) {
        try {
            await sequelize.authenticate();
            console.log(`数据库连接成功 [华为云 MariaDB] ${DB_HOST}:${DB_PORT}`);

            const models = require('./models');
            await sequelize.sync({ alter: false });
            console.log('所有模型已同步');

            dbHealthy = true;
            startHeartbeat();

            return true;
        } catch (error) {
            console.error(`数据库连接失败 (尝试 ${attempt}/${MAX_INIT_RETRIES}):`, error.message);

            if (attempt < MAX_INIT_RETRIES) {
                console.warn(`${INIT_RETRY_DELAY}ms 后重试...`);
                await new Promise(resolve => setTimeout(resolve, INIT_RETRY_DELAY));
            }
        }
    }

    console.error('数据库连接最终失败，将以无数据库模式启动');
    return false;
}

module.exports = {
    sequelize,
    initDatabase,
    checkHealth,
    startHeartbeat,
    stopHeartbeat
};

const models = require('./models');
for (const key in models) {
    module.exports[key] = models[key];
}
