const { isMockMode } = require('./mode');

let authModule;
let studentModule;
let xyyxtModule;

if (isMockMode()) {
    console.log('[Mode Adapter] 使用模拟数据模式');
    authModule = require('./api/auth');
    studentModule = require('./api/student');
    xyyxtModule = require('./xyyxt');
} else {
    console.log('[Mode Adapter] 使用生产环境模式（真实数据）');
    const originalAuth = require('./api/auth.original');
    const originalStudent = require('./api/student.original');
    const originalXyyxt = require('./xyyxt/index.original');
    
    authModule = originalAuth;
    studentModule = originalStudent;
    xyyxtModule = originalXyyxt;
}

module.exports = {
    getAuth: () => authModule,
    getStudent: () => studentModule,
    getXyyxt: () => xyyxtModule
};
