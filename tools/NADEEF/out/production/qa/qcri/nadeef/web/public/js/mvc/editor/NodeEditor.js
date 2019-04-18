/*
 * QCRI, NADEEF LICENSE
 * NADEEF is an extensible, generalized and easy-to-deploy data cleaning platform built at QCRI.
 * NADEEF means "Clean" in Arabic
 *
 * Copyright (c) 2011-2013, Qatar Foundation for Education, Science and Community Development (on
 * behalf of Qatar Computing Research Institute) having its principle place of business in Doha,
 * Qatar with the registered address P.O box 5825 Doha, Qatar (hereinafter referred to as "QCRI")
 *
 * NADEEF has patent pending nevertheless the following is granted.
 * NADEEF is released under the terms of the MIT License, (http://opensource.org/licenses/MIT).
 */

define(["d3"], function () {
    "use strict";
    var rainbow = [ "#00008B", "#8B0000", "#FF8C00", "BlueViolet", "Black", "#0000CD" ];
    var pathGenerator = d3.svg.line()
        .x(function (d) { return d.x; })
        .y(function (d) { return d.y; })
        .interpolate("basis");

    function translate(x, y) {
        return 'translate(' + (x) + ',' + (y) + ')';
    }

    function Pin(tableName, column, index) {
        this.tableName = tableName;
        this.index = index;
        this.column = column;
    }

    Pin.prototype.getName = function () {
        return this.tableName + '.' + this.column;
    };

    Table.prototype.getPoint = function (pin, leftness) {
        var x;
        var y = this.y + this.boxHeight * (pin.index + 1) + this.boxHeight * 0.5;
        if (leftness) {
            x = this.x + this.boxWidth + 10;
        } else {
            x = this.x - 10;
        }
        return { x : x, y : y };
    };

    function NodeEditor(dom, table1, table2, conns, defaultProperty) {
        this.dom = dom;
        this.svg = null;
        this.traceSvg = null;
        this.invSvg = null;
        this.conns = conns;
        this.pin0 = null;
        this.pin1 = null;
        this.table1 = _.isEmpty(table1) ? null : new Table(table1, 10, 10, this);
        this.table2 = _.isEmpty(table2) ? null : new Table(table2, 300, 10, this);
        this.defaultProperty = defaultProperty;
        this.onLineAddedHandlers = [];
    }

    NodeEditor.prototype.render = function () {
        $(this.dom).empty();
        var width = $(this.dom).width();
        var height = $(this.dom).height();

        this.svg = d3.select("#" + $(this.dom)[0].id)
            .append("svg")
            .attr("width", width)
            .attr("height", height);

        this.traceSvg = this.svg.append("g").attr("id", "trace");

        // invisible box following the mouse
        var _this = this;
        this.invSvg = this.svg.append("g").attr("class", "g_main");
        this.invSvg
            .append("rect")
            .attr("width", width)
            .attr("height", height)
            .style('visibility', 'hidden')
            .on("mousemove", function () {
                _this.redrawTrace({
                    "x" : d3.mouse(this)[0],
                    "y" : d3.mouse(this)[1]
                });
            });

        if (!_.isNull(this.table1)) {
            this.table1.render(true);
        }

        if (!_.isNull(this.table2)) {
            this.table2.render(false);
        }

        this.redrawLine();
    };

    NodeEditor.prototype.onLineAdded = function (f) {
        if (!_.isFunction(f)) {
            console.log("event handler needs to be a function");
            return;
        }

        this.onLineAddedHandlers.push(f);
    };

    NodeEditor.prototype.getPathPoints = function () {
        // path points { key: 'l1r2', points: [ x, y ] }
        var pathPoints = [];
        for (var i = 0; i < this.conns.length; i ++) {
            var conn = this.conns[i],
                p0 = this.table1.getPoint(conn.left, true),
                p3 = this.table2.getPoint(conn.right),
                p1 = { "x" : p0.x + 15, "y" : p3.y > p0.y ? p0.y + 23 : p0.y - 23},
                p2 = { "x" : p3.x - 15, "y" : p3.y > p0.y ? p3.y - 23 : p3.y + 23},
                points = [p0, p1, p2, p3],
                key = 'l' + conn.left.index + 'r' + conn.right.index;
            pathPoints.push({ "key" : key, "points" : points });
        }
        return pathPoints;
    };

    NodeEditor.prototype.redrawLine = function () {
        var pathPoints = this.getPathPoints();
        var _this = this;
        var paths =
            this.svg.selectAll("path").data(pathPoints, function (d) { return d.key; });

        paths.enter().append("path");
        paths.exit().remove();
        paths.attr("d", function (d) { return pathGenerator(d.points); })
            .attr("stroke", function (d, i) { return rainbow[i % rainbow.length]; })
            .attr("id", function (d, i) { return i; })
            .attr("stroke-width", 2)
            .attr("fill", "none")
            .on('click', function (d) {
                console.log('clicked a line');
                _.each(_this.onLineAddedHandlers, function (f) {
                    f(d);
                });
            })
            .on("mouseover", function () {
                d3.select(this).style("stroke-width", 5);
            })
            .on("mouseout", function () {
                d3.select(this).style("stroke-width", 2);
            }
        );
    };

    NodeEditor.prototype.redrawTrace = function (e) {
        if (!this.pin0 && !this.pin1)
            return;

        if (this.pin0 && this.pin1) {
            console.log("Both pins are non-null.");
            return;
        }

        var pin, leftness, table;
        if (this.pin0) {
            pin = this.pin0;
            leftness = true;
            table = this.table1;
        } else {
            pin = this.pin1;
            leftness = false;
            table = this.table2;
        }

        var traceData = [[table.getPoint(pin, leftness), { "x" : e.x, "y" : e.y }]];
        var tracePath =
            this.traceSvg
                .selectAll("path")
                .data(traceData, function() { return e.x * e.y; }
            );

        var _this = this;
        tracePath.enter().append("path");
        tracePath.exit().remove();
        tracePath
            .attr("stroke-width", 2)
            .attr("fill", "none")
            .attr("stroke",
            function() { return rainbow[_this.conns.length % rainbow.length];  }
        ).attr("d", function(d) { return pathGenerator(d); });
    };

    function Table(json, x, y, nodeEditor) {
        this.name = json.name;
        this.columns = json.columns;
        this.x = x;
        this.y = y;
        this.boxWidth = 180;
        this.boxHeight = 30;
        this.nodeEditor = nodeEditor;
    }

    Table.prototype.render = function(leftness) {
        var __ = this;
        var fontSize = 18;
        var fontOffsetX = 10;
        var fontOffsetY = fontSize;
        var drag = d3.behavior
            .drag()
            .origin(Object)
            .on("drag", function() {
                var np = d3.mouse(this.parentNode);
                __.x += np[0];
                __.y += np[1];
                __.x = Math.max(0, __.x);
                __.y = Math.max(0, __.y);
                __.x = Math.min(__.x, __.nodeEditor.dom.width() - __.boxWidth);
                __.y = Math.min(__.y, __.nodeEditor.dom.height() - __.boxHeight);
                var tableSvg = d3.select(this.parentNode);
                d3.transition(tableSvg).attr('transform', translate(__.x, __.y));
                __.nodeEditor.redrawLine();
            });

        var tableG =
            __.nodeEditor.svg.append('g')
                .attr('transform', function() {
                    return translate(__.x, __.y);
                });

        var titleG = tableG.append("g").call(drag);
        titleG.append("svg:rect")
            .style("fill", "GoldenRod")
            .style("fill-opacity", ".4")
            .style("stroke", "#777")
            .attr("width", __.boxWidth)
            .attr("height", __.boxHeight);

        titleG.append("svg:text")
            .style("font-size", fontSize)
            .style("font-family", "monospace")
            .attr("x", fontOffsetX)
            .attr("y", fontOffsetY)
            .text(__.name);

        var tableItems = tableG
            .append("g")
            .selectAll("g")
            .data(__.columns)
            .enter()
            .append('g')
            .attr("transform", function (d, i) {
                return translate(0, (i + 1) * __.boxHeight);
            })
            .attr("id", function (d, i) {
                return leftness == true ? "l" + i : "r" + i;
            })
            .on("mouseover", function () {
                d3.select(this)
                    .select("rect")
                    .style("stroke-width", 5)
                    .style("stroke", "green");
            })
            .on("mouseout", function () {
                d3.select(this)
                    .select("rect")
                    .style("stroke", "#777")
                    .style("stroke-width", 1);
            })
            .on("click", function (d) {
                if (this.id[0] === 'l') {
                    if (!_.isEmpty(__.nodeEditor.pin0)) {
                        console.log("Cannot pin the same table.");
                        __.nodeEditor.pin0 = null;
                        return;
                    }

                    var index = parseInt(this.id.substr(1));
                    __.nodeEditor.pin0 = new Pin(
                        __.nodeEditor.table1.name,
                        __.nodeEditor.table1.columns[index],
                        index
                    );
                } else {
                    if (!_.isEmpty(__.nodeEditor.pin1)) {
                        console.log("Cannot pin the same table.");
                        __.nodeEditor.pin1 = null;
                        return;
                    }
                    var index = parseInt(this.id.substr(1));
                    __.nodeEditor.pin1 = new Pin(
                        __.nodeEditor.table2.name,
                        __.nodeEditor.table2.columns[index],
                        index
                    );
                }

                if (!_.isEmpty(__.nodeEditor.pin0) &&
                    !_.isEmpty(__.nodeEditor.pin1)) {
                    __.nodeEditor.conns.push(
                        new Connection(
                            __.nodeEditor.pin0,
                            __.nodeEditor.pin1,
                            __.nodeEditor.defaultProperty()
                        )
                    );
                    __.nodeEditor.redrawLine();
                    __.nodeEditor.pin0 = null;
                    __.nodeEditor.pin1 = null;
                    _.each(__.nodeEditor.onLineAddedHandlers, function (f) { f.call(d); });
                }
            }
        );

        tableItems.append("svg:rect")
            .style("fill", "steelblue")
            .style("fill-opacity", ".4")
            .style("stroke", "#777")
            .style("stroke-width", 1)
            .attr("width", __.boxWidth)
            .attr("height", __.boxHeight);

        tableItems.append("svg:circle")
            .attr("cx", function () { return leftness ? __.boxWidth + 10 : -10; })
            .attr("cy", __.boxHeight * 0.5)
            .attr("r", 2)
            .style("fill", "none")
            .style("stroke-width", 1)
            .style("stroke", "#777");

        tableItems.append("svg:text")
            .style("font-size", fontSize)
            .style("font-family", "Aria")
            .attr("x", fontOffsetX)
            .attr("y", fontOffsetY)
            .text(function (d) { return d; });
    };

    NodeEditor.prototype.getConnection = function () {
        return this.conns;
    };

    NodeEditor.prototype.updateConnection = function (x) {
        this.conns = x;
        this.redrawLine();
    };

    NodeEditor.prototype.reset = function () {
        this.updateConnection([]);
    };

    function Connection(pin0, pin1, property) {
        this.left = pin0;
        this.right = pin1;
        this.property = property ? property : this.defaultProperty();
        this.id = 'l' + pin0.index + 'r' + pin1.index;
    }

    return {
        Create : NodeEditor,
        CreatePin : Pin,
        CreateConnection : Connection
    };
});
