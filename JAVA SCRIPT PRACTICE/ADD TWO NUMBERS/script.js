// function showText(){
//     let value=document.getElementById("box").value;
//     document.getElementById("output").innerHTML="You typed:" + value;
// // }
function addNumbers(){
    let a=Number(document.getElementById('n1').value);
    let b=Number(document.getElementById('n2').value);
    document.getElementById("sum").innerHTML="Sum is:" + (a + b);
}