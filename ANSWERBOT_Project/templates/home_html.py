<!DOCTYPE html>
<html>
<head>
    <title>
        MateM 자연어처리 테스트
    </title>
	<style type="text/css">
		a:link {color: black; text-decoration: none;}
		a:visited {color: black; text-decoration: none;}
		a:hover {color: blue; text-decoration: underline;}
	</style>
</head>
<body leftmargin="20">
    <form method="get" action="/">
        <h1> MateM 자연어처리 테스트 </h1>
        <br>
        <input type="text" name="text" style='font-size:12pt;width:500px; height:24px;padding-left:6px' autofocus>
        <button type="submit" style="width:60pt;height:24pt">입력</button>
    </form>
    <br>
    <div style='width:700px'>
    	<h3>질문:</h3>
        <p>
            {% if question != None %}
                {{question}}
            {% endif %}
		</p>
		<h3>대답:</h3>
        <p>
            {% if answer != None %}
                {{answer}}
            {% endif %}
		</p>
    </div>
    <br>
    <hr width="700" align="left">
    <br><br>
    <div>
    </div>
    <br>
    <div>
    </div>
    <br>
    <div>
    </div>
    <br>
    <div>
    </div>
    <br><br>
</html>
