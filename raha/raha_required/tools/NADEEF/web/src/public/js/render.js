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

/**
 * Render module which draws different visualization graphs.
 */
define(['table', 'requester', 'nvd3'], function (Table, Requester) {
    "use strict";
    function drawOverview(id) {
        Requester.getOverview(function (json) {
            $('#' + id + ' svg').empty();

            var data = json.data;
            // TODO: fix this hack
            var clean = data[0];
            var polluted = data[1];
            var values;
            if (clean === 0 && polluted === 0) {
                values = [];
            } else {
                var sum = clean + polluted;
                var cleanPerc = Math.round((clean / sum) * 100);
                // ceiling
                if (cleanPerc === 0 && clean !== 0) {
                    cleanPerc = 1;
                }

                var pollutedPerc = 100 - cleanPerc;

                values = [
                    {"label": "Clean Tuples " + cleanPerc + "%", "value" : clean},
                    {"label": "Dirty Tuples " + pollutedPerc + "%", "value" : polluted}
                ];
            }

            nv.addGraph(function () {
                var chart = nv.models.pieChart()
                    .x(function (d) { return d.label; })
                    .y(function (d) { return d.value; })
                    .color(d3.scale.category10().range())
                    .showLabels(true)
                    .valueFormat(d3.format('d'));

                d3.select("#" + id + " svg")
                    .datum(values)
                    .transition(10)
                    .call(chart);

                nv.utils.windowResize(chart.update);
                return chart;
            });
        });
    }

    function drawAttribute(id) {
        Requester.getAttribute({
            success: function (json) {
                $('#' + id + ' svg').empty();
                var result = json.data;
                var values;
                if (result == null && result.length === 0) {
                    values = null;
                } else {
                    values = [];
                    for (var i = 0; i < result.length; i++) {
                        values.push(
                            {'label': result[i][0], 'value': result[i][1]}
                        );
                    }
                }

                var graphData = [{
                    key: "Cumulative Return",
                    values: values
                }];

                nv.addGraph(function () {
                    var chart = nv.models.discreteBarChart()
                        .x(function (d) {
                            return d.label;
                        })
                        .y(function (d) {
                            return d.value;
                        })
                        .staggerLabels(false)
                        .tooltips(true)
                        .showValues(true)
                        .valueFormat(d3.format('d'));

                    chart.xAxis.axisLabel("Attribute");
                    chart.yAxis
                        .axisLabel("Number of Violation")
                        .axisLabelDistance(50)
                        .tickFormat(d3.format('d'));
                    d3.select("#" + id + " svg")
                        .datum(graphData)
                        .transition().duration(10)
                        .call(chart);

                    nv.utils.windowResize(chart.update);

                    chart.discretebar.dispatch.on('elementClick', function (e) {
                        console.log(e.point.label);
                        Table.filter(e.point.label);
                    });
                    return chart;
                });
            }
        });
    }

    var svg;        // outer svg
    var link;       // svg links
    var node;       // svg nodes
    var clusterMap; // node cluster map;
    function drawViolationRelation(id) {
        Requester.getViolationRelation({
            success: function (data) {
                var width = $('#' + id).width();
                var height = $('#' + id).height();

                $("#" + id + " svg").empty();
                var result = data.data;
                // TODO: find a better pattern
                if (data.code === 2) {
                    d3.select("#" + id + " svg")
                        .append("text")
                        .attr("x", width / 2 - 80)
                        .attr("y", height / 2 - 50)
                        .style("font-size", "20px")
                        .style("font-weight", "bold")
                        .text("There are too many violations to show.");
                    return;
                }

                if (data.data && data.data.length === 0) {
                    d3.select("#" + id + " svg")
                        .append("text")
                        .attr("x", width / 2 - 80)
                        .attr("y", height / 2 - 50)
                        .style("font-size", "20px")
                        .style("font-weight", "bold")
                        .text("No Data Available.");
                    return;
                }

                var i,
                    j,
                    k,
                    tmp,
                    graph = {},
                    nodes = [],
                    nodeTable = {},
                    tableHash = {},
                    buildId = function (d) { return d[2] + '_' + d[1]; };

                // first loop to find distinct nodes
                for (i = 0; i < result.length; i ++) {
                    tmp = buildId(result[i]);
                    if (!(tmp in graph)) {
                        graph[tmp] = {};
                        var tableName = result[i][2];
                        var tableId;
                        if (tableName in tableHash) {
                            tableId = tableHash[tableName];
                        } else {
                            var keys = Object.keys(tableHash);
                            tableId = keys.length + 1;
                            tableHash[tableName] = tableId;
                        }
                        var newNode = {
                                'name': tmp,
                                'group': tableId,
                                'tableName' : tableName,
                                'tid' : result[i][1]
                            };
                        nodeTable[tmp] = newNode;
                        nodes.push(newNode);
                    }
                }

                // second loop to create graph matrix
                var preVid = null,
                    cc = [];

                for (i = 0; i < result.length; i ++) {
                    var vid = result[i][0];
                    if (vid === preVid) {
                        cc.push(result[i]);
                    } else {
                        for (j = 0; j < cc.length; j ++) {
                            var id1 = buildId(cc[j]);
                            for (k = j + 1; k < cc.length; k ++) {
                                var id2 = buildId(cc[k]);
                                graph[id1][id2] = 1;
                                graph[id2][id1] = 1;
                            }
                        }
                        cc = [result[i]];
                        preVid = vid;
                    }
                }

                // third loop to create edges and clusters
                var tuples = Object.keys(graph),
                    links = [],
                    linkSet = {};

                clusterMap = {};
                for (i = 0; i < tuples.length; i ++) {
                    var components = graph[tuples[i]];
                    var targets = Object.keys(components);
                    var sourceId = tuples[i];
                    var cluster;

                    // create new cluster if not exists
                    if (sourceId in clusterMap) {
                        cluster = clusterMap[sourceId];
                    } else {
                        cluster = [nodeTable[sourceId]];
                        clusterMap[sourceId] = cluster;
                    }

                    for (j = 0; j < targets.length; j ++) {
                        var targetId = targets[j];
                        var hash0 = sourceId + '_' + targetId;
                        var hash1 = targetId + '_' + sourceId;
                        if (!(hash0 in linkSet || hash1 in linkSet)) {
                            links.push({
                                'source': nodeTable[sourceId],
                                'target': nodeTable[targetId],
                                'value': 1,
                                'weight': 1
                            });

                            linkSet[hash0] = 1;
                            linkSet[hash1] = 1;
                        }

                        // merge two existing cluster
                        if (targetId in clusterMap) {
                            var mergeCluster = clusterMap[targetId];
                            if (mergeCluster !== cluster) {
                                if (targetId === 'tb_amazon_560') {
                                    console.log('got it');
                                }

                                cluster = cluster.concat(mergeCluster);
                                // assign new cluster
                                for (k = 0; k < cluster.length; k ++) {
                                    clusterMap[cluster[k].name] = cluster;
                                }
                            }
                        } else {
                            // adds to the existing cluster
                            cluster.push(nodeTable[targetId]);
                            clusterMap[targetId] = cluster;
                        }
                    }
                }

                // render time
                var container = $("#" + id);
                var color = d3.scale.category10();
                svg = d3.select("#" + id + " svg")
                    .attr('width', container.width())
                    .attr('height', container.height())
                    .append('g')
                    .call(d3.behavior.zoom().scaleExtent([0.1, 20]).on("zoom", rescale));

                svg.append('svg:rect')
                    .attr('width', container.width())
                    .attr('height', container.height())
                    .attr("fill", "none")
                    .attr("pointer-events", "all")
                    .append("g");

                var force = d3.layout.force()
                    .charge(-150)
                    .linkDistance(50)
                    .size([container.width(), container.height()])
                    .nodes(nodes)
                    .links(links)
                    .start();

                link = svg.selectAll(".link")
                    .data(links)
                    .enter()
                    .append("line")
                    .style("stroke", "steelblue")
                    .style("stroke-width", function () { return 2.0; });

                node = svg.selectAll(".node")
                    .data(nodes)
                    .enter()
                    .append("circle")
                    .attr("r", function (d) {
                        var components = graph[d.name];
                        return Object.keys(components).length * 2 + 8;
                    }).style("fill", function (d) { return color(d.group % 20); })
                    .on('click', function (d) {
                        Table.filterByCluster(clusterMap[d.name]);
                    }).call(force.drag);

                node.append("title")
                    .text(function (d) { return d.tableName + ':' + d.tid; });

                force.on("tick", function () {
                    link.attr("x1", function (d) { return d.source.x; })
                        .attr("y1", function (d) { return d.source.y; })
                        .attr("x2", function (d) { return d.target.x; })
                        .attr("y2", function (d) { return d.target.y; });

                    node.attr("cx", function (d) { return d.x; })
                        .attr("cy", function (d) { return d.y; });
                });
            }
        });
    }

    function rescale() {
        link.attr(
            "transform",
            "translate(" + d3.event.translate + ") scale(" + d3.event.scale + ")"
        );

        node.attr(
            "transform",
            "translate(" + d3.event.translate + ") scale(" + d3.event.scale + ")"
        );
    }

    function drawDistribution(id) {
        Requester.getRuleDistribution(function (data) {
            $('#' + id + '  svg').empty();
            var result = data.data;
            if (result == null || result.length === 0) {
                return;
            }

            var affectedTuple = [];
            var affectedTable = [];
            for (var i = 0; i < result.length; i ++) {
                affectedTuple.push({'label' : result[i][0], 'value' : result[i][1]});
                affectedTable.push({'label' : result[i][0], 'value' : result[i][2]});
            }

            var distributionData = [
                {
                    key: 'Affected Tuple',
                    color: '#d62728',
                    values: affectedTuple
                },
                {
                    key: 'Affected Table',
                    color: '#1f77b4',
                    values: affectedTable
                }
            ];

            nv.addGraph(function () {
                var chart = nv.models.multiBarHorizontalChart()
                    .x(function (d) { return d.label; })
                    .y(function (d) { return d.value; })
                    .showValues(true)
                    .tooltips(true)
                    .showControls(false)
                    .valueFormat(d3.format('d'));

                chart.yAxis
                    .tickFormat(d3.format('d'))
                    .axisLabel("Number of violation");

                d3.select("#" + id + " svg")
                    .datum(distributionData)
                    .transition(10)
                    .call(chart);

                nv.utils.windowResize(chart.update);

                chart.dispatch.on(
                    'stateChange',
                    function (e) { nv.log('New State:', JSON.stringify(e)); }
                );

                return chart;
            });
        });
    }

    function drawTupleRank(id) {
        Requester.getTupleRank(function (data) {
            $('#' + id + ' svg').empty();
            var result = data.data;
            var values = [];
            for (var i = 0; i < result.length; i ++) {
                values.push({'label' : result[i][0], 'value' : result[i][1]});
            }

            var graphData = [{
                key: "Tuple Rank",
                values: values
            }];

            nv.addGraph(function () {
                var chart = nv.models.discreteBarChart()
                    .x(function (d) { return "TID: " + d.label; })
                    .y(function (d) { return d.value; })
                    .staggerLabels(false)
                    .tooltips(true)
                    .showValues(false)
                    .valueFormat(d3.format('d'));

                chart.yAxis
                    .tickFormat(d3.format('d'))
                    .axisLabelDistance(50)
                    .axisLabel("Number of involved violation");

                chart.showXAxis(false);

                var container = $("#" + id);
                d3.select("#" + id + " svg")
                    .attr("width", container.width())
                    .attr("height", container.height())
                    .datum(graphData)
                    .transition().duration(10)
                    .call(chart);

                d3.select("#" + id + " svg")
                    .append("text")
                    .attr("x", container.width() / 2 - 100)
                    .attr("y", container.height() - 20)
                    .style("weight", "bold")
                    .text("Violation rank by tuple");

                nv.utils.windowResize(chart.update);

                chart.discretebar.dispatch.on('elementClick', function (e) {
                    Table.filter("?=" + e.point.label);
                });
                return chart;
            });
        });
    }

    return {
        drawOverview : drawOverview,
        drawAttribute : drawAttribute,
        drawDistribution : drawDistribution,
        drawTupleRank : drawTupleRank,
        drawViolationRelation : drawViolationRelation
    };
});
