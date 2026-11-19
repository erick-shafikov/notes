<!DOCTYPE html>
<script>

function func(){
    let finalArr = [];

    function recFunc(arr){
        arr.forEach(el => {
            Array.isArray(el) ? recFunc(el) : finalArr.push(el)
        })
    }
    
    recFunc(Array.from(arguments));

    return finalArr.reduce((a, b) => a + b, 0);
}

console.log(func(1,2,[[3,4],5],6));



//1,2,3,[4,5],6
//1,2,[[3,4],5],6
</script>