// alert();
// function hello(){
//     alert('hello');
//     alert('dd');
// }

// function go(name, age){
//     alert()
// }

// function add(a, b){
//     return a+b;
// }

// function multi(a, b){
//     return a*b;
// }
// var a = prompt('输入乘数1');
// var b = prompt('输入乘数2');
// alert('相乘结果为:'+multi(a,b));

// var time = 0;

// while (time<10){
// console.log('尝试了',time,'次');
// time++;
// }
// var myList = ['dd', 'dyx', 'sb']

// for (var i=0; i < myList.length; i++){
//     console.log('i is', myList[i]);
// }

var numOne = document.getElementById("num-one")
var numTwo = document.getElementById("num-two")
var addSum = document.getElementById("add-sum")

numOne.addEventListener("input", add);
numTwo.addEventListener("input", add);
function add() {
    var one = parseFloat(numOne.value)||0;
    // 防止出现null,undefined
    var two = parseFloat(numTwo.value)||0;
    var sum = one+two;
    // 方便显示
    addSum.innerHTML = '两数之和为: ' + (one+two);
    // 改变addSum的内部值
    // 用括号括起来,先运算后面的加法,在转换成字符串拼接
}
