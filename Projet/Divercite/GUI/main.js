//wait for document to be ready
$(document).ready(function() {

    /* Board representation */
    const canvasBoard = document.getElementById("board");
    const ctxBoard = canvasBoard.getContext("2d");
    const gridSize = 9;
    const margin = 0.12 * canvasBoard.width;
    const interPieceMargin = canvasBoard.width * 0.032;
    const cellSize = (canvasBoard.width - 2 * margin) / gridSize;
    const centerX = canvasBoard.width / 2;
    const centerY = canvasBoard.height / 2;
    const boardPath = './assets/Plateau.webp';
    const pieceBgPath = './assets/piece_bg.webp';

    /* Player representation */
    const canvasPlayers = {
        'W': document.getElementById("player1"),
        'B': document.getElementById("player2")
    };
    const ctxPlayers = {
        'W': canvasPlayers['W'].getContext("2d"),
        'B': canvasPlayers['B'].getContext("2d")
    };

    // Load the piece images
    const pieceImages = {};
    const pieceTypes = ['R', 'C'];
    const pieceColors = ['R','G','B','Y'];
    const playerColor = ['B', 'W'];
    for (const type of pieceTypes) {
        for(const color of pieceColors){
            for (const player of playerColor) {
                const pieceImg = new Image();
                if (type === 'R') {
                    pieceImg.src = `./assets/${color}${type}.png`;
                    pieceImages[`${color}${type}`] = pieceImg;
                    break;
                }
                else {
                    pieceImg.src = `./assets/${color}${type}${player}.png`;
                    pieceImages[`${color}${type}${player}`] = pieceImg;
                }
            }
        }
    }
    // background image for the pieces
    const pieceBgImg = new Image();
    pieceBgImg.src = pieceBgPath;
    
    const BOARD_MASK = [
        [ 0,   0,   0,   0,  'R',  0,   0,   0,   0],
        [ 0,   0,   0,  'R', 'C', 'R',  0,   0,   0],
        [ 0,   0,  'R', 'C', 'R', 'C', 'R',  0,   0],
        [ 0,  'R', 'C', 'R', 'C', 'R', 'C', 'R',  0],
        ['R', 'C', 'R', 'C', 'R', 'C', 'R', 'C', 'R'],
        [ 0,  'R', 'C', 'R', 'C', 'R', 'C', 'R',  0],
        [ 0,   0,  'R', 'C', 'R', 'C', 'R',  0,   0],
        [ 0,   0,   0,  'R', 'C', 'R',  0,   0,   0],
        [ 0,   0,   0,   0,  'R',  0,   0,   0,   0]
    ]

    const cellCoordinates = {};
    for (let row = 0; row < gridSize; row++) {
        for (let col = 0; col < gridSize; col++) {
            const cellCenterCoordinates = getCellCoordinates(row, col);
            cellCoordinates[`${row},${col}`] = cellCenterCoordinates;
        }
    }
    const firstPlayersData = {
        "W": {
            "name": "Player 1",
            "score": 0,
            "pieces_left": {
                "RC": 2,
                "RR": 3,
                "GC": 2,
                "GR": 3,
                "BC": 2,
                "BR": 3,
                "YC": 2,
                "YR": 3,
            },
            "id": 1,
        },
        "B": {
            "name": "Player 2",
            "score": 0,
            "pieces_left": {
                "RC": 2,
                "RR": 3,
                "GC": 2,
                "GR": 3,
                "BC": 2,
                "BR": 3,
                "YC": 2,
                "YR": 3,
            },
            "id": 2,
        }
            
    };

    // Load and draw the background board
    const boardImg = new Image();
    boardImg.src = boardPath;
    // wait for the image to be loaded before proceeding
    boardImg.onload = function() {
        drawGrid({});
        drawPlayersRepresentation(firstPlayersData);

    }
    var piecesLeftCoordinates = {};
    var steps = [];
    var index = -1;
    var play = false;
    var lastCellMouseOn = null;
    var lastPieceMouseOn = null;
    var selectedPiece = null;
    var next_player;
    var loop = null;
    var playersName = {
        "W": "Player 1",
        "B": "Player 2"
    };
    var socket = io({
        reconnection: false,
    });

    
    for (const player in canvasPlayers) {
        canvasPlayers[player].addEventListener('mousemove', function(event) {
            const rect = canvasPlayers[player].getBoundingClientRect();
            const mouseX = event.clientX - rect.left;
            const mouseY = event.clientY - rect.top;
            handleMouseMovePlayer(mouseX, mouseY, player);
        });
        canvasPlayers[player].addEventListener('click', function(event) {
            const rect = canvasPlayers[player].getBoundingClientRect();
            const mouseX = event.clientX - rect.left;
            const mouseY = event.clientY - rect.top;
            handleMouseClickPlayer(mouseX, mouseY, player);
        });
    }
    
    canvasBoard.addEventListener('mousemove', function(event) {
        const rect = canvasBoard.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;
        handleMouseMove(mouseX, mouseY);
    });
    canvasBoard.addEventListener('click', function(event) {
        const rect = canvasBoard.getBoundingClientRect();
        const mouseX = event.clientX - rect.left;
        const mouseY = event.clientY - rect.top;
        handleMouseClickBoard(mouseX, mouseY);
    });

    function handleMouseClickBoard(mouseX, mouseY) {
        const cell = isMouseOnCell(mouseX, mouseY);
        if (cell !== null && selectedPiece !== null) {
            socket.emit("interact", JSON.stringify({
                "piece": selectedPiece,
                "position": cell,
            }));
            selectedPiece = null;
            lastPieceMouseOn = null;
            lastCellMouseOn = null;
        }
    }

    function handleMouseClickPlayer(mouseX, mouseY, player) {
        if (playersName[player] !== next_player) return;
        const piece = isMouseOnPiece(mouseX, mouseY);
        if (piece !== null) {
            if (piece !== selectedPiece) {
                selectedPiece = piece;
                // make a highlight on the selected piece
                ctxPlayers[player].clearRect(0, 0, canvasPlayers[player].width, canvasPlayers[player].height);
                drawPlayersRepresentation(steps[index] ? steps[index].players : firstPlayersData);
                drawPiece(piecesLeftCoordinates[piece], ctxPlayers[player], pieceBgImg, false);
                highlightPiece(piece, ctxPlayers[player]);
            }
            else {
                selectedPiece = null;
                ctxPlayers[player].clearRect(0, 0, canvasPlayers[player].width, canvasPlayers[player].height);
                drawPlayersRepresentation(steps[index] ? steps[index].players : firstPlayersData);
                drawPiece(piecesLeftCoordinates[piece], ctxPlayers[player], pieceBgImg, false);
            }
        }
    }

            

    function handleMouseMovePlayer(mouseX, mouseY, player) {
        if (playersName[player] !== next_player) return;
        const piece = isMouseOnPiece(mouseX, mouseY);
        if (piece !== null && (lastPieceMouseOn === null || piece+player !== lastPieceMouseOn)) {
            canvasPlayers[player].style.cursor = "pointer";
            ctxPlayers[player].clearRect(0, 0, canvasPlayers[player].width, canvasPlayers[player].height);
            drawPlayersRepresentation(steps[index] ? steps[index].players : firstPlayersData);
            drawPiece(piecesLeftCoordinates[piece], ctxPlayers[player], pieceBgImg, false);
            highlightPiece(selectedPiece, ctxPlayers[player]);
            lastPieceMouseOn = piece+player;
        }else if(piece === null){
            canvasPlayers[player].style.cursor = "default";
        }
    }


    function highlightPiece(piece, ctx) {
        // highlight the piece
        if (piece === null) return;
        ctx.beginPath();
        ctx.lineWidth = 3;
        ctx.strokeStyle = "red";
        ctx.arc(piecesLeftCoordinates[piece].x, piecesLeftCoordinates[piece].y, cellSize / 2, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.closePath();
    }

    function isMouseOnPiece(mouseX, mouseY) {
        for (const key in piecesLeftCoordinates) {
            const x = piecesLeftCoordinates[key].x;
            const y = piecesLeftCoordinates[key].y;
            const distance = Math.sqrt((mouseX - x) ** 2 + (mouseY - y) ** 2);
            if (distance < cellSize / 2) {
                return key;
            }
        }
        return null;
    }

    function handleMouseMove(mouseX, mouseY) {
        const cell = isMouseOnCell(mouseX, mouseY);
        if (cell !== null && (lastCellMouseOn === null || cell.toString() !== lastCellMouseOn.toString())) {
            canvasBoard.style.cursor = "pointer";
            ctxBoard.clearRect(0, 0, canvasBoard.width, canvasBoard.height);
            drawGrid(steps[index] ? steps[index].env : {});
            placePiece(cell[0], cell[1], pieceBgImg);
            lastCellMouseOn = cell;
        }else if(cell === null){
            canvasBoard.style.cursor = "default";
          }
    }

    function isMouseOnCell(mouseX, mouseY){
        for (const key in cellCoordinates) {
            const x = cellCoordinates[key].x;
            const y = cellCoordinates[key].y;
            const distance = Math.sqrt((mouseX - x) ** 2 + (mouseY - y) ** 2);
            if (distance < cellSize / 2) {
                const [row, col] = key.split(',').map(Number);
                if (BOARD_MASK[row][col] !== 0) {
                    return [row, col];
                }
            }        
        }
        return null;
    }

    
    $("#loadJson").on("change", function() {
        const file = this.files[0];
        const reader = new FileReader();
        reader.onload = function(e) {
            const json = JSON.parse(e.target.result);
            let gameData = [];
            for (const step of json) {
                const players_info = convertToPlayerInfo(step.players, step.scores, step.players_pieces_left);
                gameData.push({"env": step.rep.env, "players":players_info, "next_player": step.next_player.name});
            }
            steps = gameData;
            index = 0;
            drawNewState(steps[index]);
        }
        reader.readAsText(file);
    });


    $('#time').on('change', function() {
        play = false;
        $("#play").click();
    });

    $("#play").click(function() {
        play = true;
        
        if (loop) {
            clearInterval(loop);
        }
        
        loop = setInterval(function() {
            if (play) {
                if (index < steps.length - 1) {
                    index++;
                    drawNewState(steps[index]);
                } else {
                    play = false;
                    clearInterval(loop);
                }
            } else {
                clearInterval(loop);
            }
        }, $("#time").attr("max") - $("#time").val());
    });

    $("#stop").click(function() {
        play = false;
    });

    $("#reset").click(function() {
        index = 0;
        drawNewState(steps[index]);
    });
    $("#next").click(function() {
        if (index < steps.length - 1) {
            index++;
            drawNewState(steps[index]);
        }
    });

    $("#previous").click(function() {
        if (index > 0) {
            index--;
            drawNewState(steps[index]);
        }
    });
    $("#close_pop_up").click(function() {
        $("#pop_up_container").css("display", "none");

    });
    connect_handler = () => {
        socket = io("ws://" + $("#hostname")[0].value + ":" + $("#port")[0].value + "", {
            reconnection: false,
        });

        socket.on("connect_error", (err) => {
            $("#connect").addClass("connection_error");

        });

        socket.on("connect", () => {
            $("#connect").removeClass("connection_error");
            socket.emit("identify", JSON.stringify({
                "identifier": "__GUI__" + Date.now()
            }));
            $("#status")[0].innerHTML = 'Connected';
            $("#status")[0].style = 'color:green';
            $("#connect").unbind();
            $('#loadJsonButton').addClass('disabled');
            
        });

        socket.on("play", (...args) => {
            console.log("play");
            json = JSON.parse(args[0]);
            if (!json.rep) json = JSON.parse(json);
            if (json.rep && json.rep.env) {
                nextPlayer = json.next_player.name;
                const players_info = convertToPlayerInfo(json.players, json.scores, json.players_pieces_left);
                steps.push({"env": json.rep.env, "players":players_info, "next_player": nextPlayer});
                index = steps.length - 1;
                drawNewState(steps[index]);
            }
        });

        socket.on("ActionNotPermitted", (...args) => {
            // TODO: Display message to user
            $("#error").css("opacity", "1");
            setTimeout(function() {
                $("#error").css("opacity", "0");
            }, 2000);
            selectedPiece = null;
            drawPlayersRepresentation(steps[index] ? steps[index].players : firstPlayersData);
        })

        socket.on("disconnect", (...args) => {
            // set display block to img of id #img
            $("#status")[0].innerHTML = 'Disconnected';
            $("#status")[0].style = 'color:red';
            $("#connect").click(connect_handler);
            $('#loadJsonButton').removeClass('disabled');
        });

        socket.on("done", (...args) => {
            score = JSON.parse(args[0])

            winner_id = ""
            winner_score = -10
            for (p in score) {
                if (score[p] > winner_score) {
                    winner_id = p
                    winner_score = score[p]
                }
            }

            loser_score = winner_score

            for (p in score) {
                if (score[p] != loser_score) {
                    loser_score = score[p]
                    break;
                }
            }

            if (winner_score == loser_score) {
                text = "Match nul ! Scores : " + winner_score + " - " + loser_score
            } else {
                winner_name = ""
                winner_color = ""
                for (p in steps[steps.length - 1].players) {
                    if (steps[steps.length - 1].players[p].id == winner_id) {
                        winner_name = steps[steps.length - 1].players[p].name
                        winner_color = p
                    }
                }
                text = "Le joueur " + winner_name + " (<span class=\"winner_indicator " + winner_color + "\"></span>) est vainqueur avec " + winner_score + " points contre " + loser_score
            }
            $("#pop_up_container").css("display", "flex");
            $("#pop_up_text").html(text);

        })

    }

    $("#connect").click(connect_handler);
    connect_handler();

    function drawNewState(newState) {
        if (newState === undefined){
            newState = {"env": {}, "players": firstPlayersData, "next_player": "White"};
        }
        next_player = newState.next_player;
        drawGrid(newState.env);
        drawPlayersRepresentation(newState.players);
        
    }

    function convertToPlayerInfo(players, scores, playersPiecesLeft){
        let playerInfo = {};
        for (let player of players) {
            let playerColor = player.piece_type;
            playerInfo[playerColor] = {
                "name": player.name,
                "score": scores[player.id],
                "pieces_left": playersPiecesLeft[player.id],
                "id": player.id,
            }
            playersName[playerColor] = player.name;
        }
        return playerInfo;
    }

    function drawPlayersRepresentation(playerData) { 
        if (playerData === undefined) {
            playerData = firstPlayersData;
        }   
        for (const player in playerData) {
            const ctx = ctxPlayers[player];
            ctx.clearRect(0, 0, canvasPlayers[player].width, canvasPlayers[player].height);
    
            // Draw background frame
            // ctx.fillStyle = "#ffdef5"; // Light gray background
            // ctx.fillRect(0, 0, canvasPlayers[player].width, canvasPlayers[player].height);
    
            //ctx.strokeStyle = "#333"; // Dark border
            //ctx.lineWidth = 4;
            //ctx.strokeRect(0, 0, canvasPlayers[player].width, canvasPlayers[player].height);
            // Draw player details
            ctx.font = "bold 20px Arial";
            ctx.fillStyle = "#333"; // Dark text color
            // Calculate text width for centering
            const playerNameText = playerData[player].name;
            const playerScoreText = `Score  ${playerData[player].score}`;
            const playerNameTextWidth = ctx.measureText(playerNameText).width;
            const playerScoreTextWidth = ctx.measureText(playerScoreText).width;

            const canvasCenterX = canvasPlayers[player].width / 2;

            // Draw centered text
            ctx.fillText(playerNameText, canvasCenterX - playerNameTextWidth / 2, 80);
            ctx.fillText(playerScoreText, canvasCenterX - playerScoreTextWidth / 2, 140);
            // Frame the text 
            if (playerData[player].name === next_player) {
                ctx.strokeStyle = "#333"; // Dark border
                ctx.lineWidth = 2;
                ctx.strokeRect(canvasCenterX - playerNameTextWidth / 2 - 10, 50, playerNameTextWidth + 20, 40);
            }

            // Draw the pieces left
            let yPosition = ctx.canvas.height / 4;
            let xPosition = ctx.canvas.width / 5;
            let i = 0;
            for (let [pieceType, count] of Object.entries(playerData[player].pieces_left)) {
                if (pieceType[1] === 'C') pieceType = pieceType + player;
                const pieceImg = pieceImages[pieceType];
                
                if (i % 2 === 0) {
                    xPosition = ctx.canvas.width / 5;
                    yPosition += cellSize + 20; // Increased space between rows
                }
                piecesLeftCoordinates[pieceType.substring(0, 2)] = { x: xPosition + cellSize / 2, y: yPosition + cellSize / 2 };
    
                if (!pieceImg.complete || pieceImg.naturalWidth === 0) {
                    pieceImg.onload = function() {
                        ctx.drawImage(pieceImg, xPosition, yPosition, cellSize, cellSize);
                    };
                } else {
                    ctx.drawImage(pieceImg, xPosition, yPosition, cellSize, cellSize);
                }
    
                // Draw the count of pieces left
                ctx.font = "bold 20px Arial";
                ctx.fillStyle = "#333"; // Dark text color
                ctx.fillText(`${count}`, xPosition + cellSize + 15, yPosition + cellSize / 2 + 5);
    
                // Draw border around the piece
                ctx.strokeStyle = "#000"; // Black border
                ctx.lineWidth = 2;
                ctx.strokeRect(xPosition, yPosition, cellSize, cellSize);
    
                xPosition += cellSize * 3;
                i++;
            }
        }
    }
    
    function drawGrid(rep_env) {
        ctxBoard.clearRect(0, 0, canvasBoard.width, canvasBoard.height);
        ctxBoard.drawImage(boardImg, 0, 0, canvasBoard.width, canvasBoard.height);
        // frame the board
        ctxBoard.strokeStyle = "#000"; // Dark border
        ctxBoard.lineWidth = 15;
        ctxBoard.strokeRect(0, 0, canvasBoard.width, canvasBoard.height);

        //keys of event are (6,6) : {piece_type: 'R', owner_id: 1}
        for (const key in rep_env) {
            const [row, col] = key.split("(")[1].split(")")[0].split(",").map(Number);
            let pieceInfos = rep_env[key].piece_type;
            let pieceImg;
            if (pieceInfos[1] === 'R') pieceInfos = pieceInfos.substring(0, 2);
            pieceImg = pieceImages[pieceInfos];
            placePiece(row, col, pieceImg);
        }
    }

    

    function placePiece(row, col, pieceImg) {
        if (!pieceImg.complete || pieceImg.naturalWidth === 0) {
            pieceImg.onload = function() {
                drawPiece(getCellCoordinates(row, col), ctxBoard, pieceImg);
            };
        } else {
            drawPiece(getCellCoordinates(row, col),ctxBoard, pieceImg);
        }
    }
    
    function drawPiece(coord, ctx, pieceImg, rotate = true) {
        if (rotate) {
            ctx.save();
            ctx.translate(coord.x, coord.y);
            ctx.rotate(Math.PI / 4);
            ctx.drawImage(pieceImg, -cellSize / 2, -cellSize / 2, cellSize, cellSize);
            ctx.restore();
        }
        else {
            ctx.drawImage(pieceImg, coord.x - cellSize / 2, coord.y - cellSize / 2, cellSize, cellSize);
        }
    }

    function calculateRotatedCoordinates(x, y, angle) {
        // compute the new coordinates after rotation where the origin is the center of the canvasBoard
        const newX = centerX + (x - centerX) * Math.cos(angle) - (y - centerY) * Math.sin(angle);
        const newY = centerY + (x - centerX) * Math.sin(angle) + (y - centerY) * Math.cos(angle);
        return { x: newX, y: newY };
    }

    function getCellCoordinates(row, col) {
        // center of the cell
        const x = centerX + (col - 4) * (cellSize + interPieceMargin);
        const y = centerY + (row - 4) * (cellSize + interPieceMargin);
        const coord = calculateRotatedCoordinates(x, y, Math.PI / 4);
        return { x: coord.x, y: coord.y };
    }

    function drawPoint(x, y) {
        ctxBoard.beginPath();
        ctxBoard.arc(x, y, 3, 0, 2 * Math.PI);
        ctxBoard.fillStyle = 'red';
        ctxBoard.fill();
        ctxBoard.closePath();
    }
});
// socket.emit("interact", JSON.stringify({
//                           "from": getLocation(activeBall),
//                           "to": getLocation(secondLoc),
//                           "type": type
//                       }));
