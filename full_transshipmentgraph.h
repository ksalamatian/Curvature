/* -*- mode: C++; indent-tabs-mode: nil; -*-
 * This file has been adapted by Nicolas Papadakis (2018) for
 transhipment problems.
 * It is based on the adaptation of full_bipartitegraph.h by Nicolas Bonneel (2013),
 * from full_graph.h from LEMON, a generic C++ optimization library,
 * to implement a lightweight fully connected bipartite graph. A previous
 * version of this file is used as part of the Displacement Interpolation
 * project,
 * Web: http://www.cs.ubc.ca/labs/imager/tr/2011/DisplacementInterpolation/
 *
 *
 **** Original file Copyright Notice :
 * Copyright (C) 2003-2010
 * Egervary Jeno Kombinatorikus Optimalizalasi Kutatocsoport
 * (Egervary Research Group on Combinatorial Optimization, EGRES).
 *
 * Permission to use, modify and distribute this software is granted
 * provided that this copyright notice appears in all copies. For
 * precise terms see the accompanying LICENSE file.
 *
 * This software is provided "AS IS" with no warranty of any kind,
 * express or implied, and with no claim as to its suitability for any
 * purpose.
 *
 */

#ifndef LEMON_FULL_TRANSSHIPMENT_GRAPH_H
#define LEMON_FULL_TRANSSHIPMENT_GRAPH_H
#include <cstdint>
//#include "core.h"

///\ingroup graphs
///\file
///\brief FullTransShipmentgraph class.


namespace lemon {
#define TSGRAPH_TYPEDEFS(TSgraph)                                       \
typedef TSgraph::Node Node;                                           \
typedef TSgraph::Arc Arc;                                             \


#define TEMPLATE_TSGRAPH_TYPEDEFS(TSgraph)                              \
typedef typename TSgraph::Node Node;                                  \
typedef typename TSgraph::Arc Arc;                                    \


    class FullTransShipmentgraphBase {
    public:

        typedef FullTransShipmentgraphBase TSgraph;

        //class Node;
        typedef int Node;
        //class Arc;
        typedef int64_t Arc;

    protected:

        int _node_num;
        int64_t _arc_num;

        FullTransShipmentgraphBase() {}



        void construct(int n1, int n_lay1, int n2) {
            _node_num = n1+n2+n_lay1;
            _arc_num = ((int64_t)n1 +(int64_t)n2)*(int64_t)n_lay1;
            _n1=n1;
            _n_layer1=n_lay1;
            _n2=n2;
            _index_layer1=(int64_t)_n1*(int64_t)n_lay1;
            _end_layer1=_n1+n_lay1;
        }



    public:

        int _n1, _n2,_n_layer1,_end_layer1;
        int64_t _index_layer1;

        Node operator()(int ix) const { return Node(ix); }
        static int index(const Node& node) { return node; }

        Arc arc(const Node& s, const Node& t) const {
            if (s<_n1 && t>=_n1 && t<_end_layer1)
                return Arc((int64_t)s * (int64_t)_n_layer1 + (int64_t)(t-_n1) );
            else {
                if (s>=_n1 && s<_end_layer1 && t>=_end_layer1)
                    // return Arc((t+_n1) * _n_layer1 + (s-_n_layer1+_n1) );
                    return Arc(_index_layer1+(int64_t)(s-_n1)*(int64_t)_n2+(int64_t)(t-_end_layer1));
                else
                    return Arc(-1);
            }



        }

        void reset(int n1, int n_lay1, int n2) {
            _node_num = n1+n2+n_lay1;
            _arc_num = ((int64_t)n1 +(int64_t)n2)*(int64_t)n_lay1;
            _n1=n1;
            _n_layer1=n_lay1;
            _n2=n2;
            _index_layer1=(int64_t)_n1*(int64_t)n_lay1;
            _end_layer1=_n1+n_lay1;
        }

        int nodeNum() const { return _node_num; }
        int64_t arcNum() const { return _arc_num; }

        int maxNodeId() const { return _node_num - 1; }
        int64_t maxArcId() const { return _arc_num - 1; }

        Node source(Arc arc) const {
            if (arc <_index_layer1)
                return arc / _n_layer1;
            else
//                   return (arc % _n_layer1) + _n1;
                return (arc-_index_layer1) / _n2+_n1;




        }
        Node target(Arc arc) const {

            if (arc <_index_layer1)
                return (arc % _n_layer1) + _n1;
            else
//                    return arc / _n_layer1+_n_layer1;
                return ((arc-_index_layer1) % _n2) + _end_layer1;


        }

        static int id(Node node) { return node; }
        static int64_t id(Arc arc) { return arc; }

        static Node nodeFromId(int id) { return Node(id);}
        static Arc arcFromId(int64_t id) { return Arc(id);}


        Arc findArc(Node s, Node t, Arc prev = -1) const {
            return prev == -1 ? arc(s, t) : -1;
        }

        void first(Node& node) const {
            node = _node_num - 1;
        }

        static void next(Node& node) {
            --node;
        }

        void first(Arc& arc) const {
            arc = _arc_num - 1;
        }

        static void next(Arc& arc) {
            --arc;
        }

        void firstOut(Arc& arc, const Node& node) const {
            if (node>=_end_layer1)
                arc = -1;
            else
            if (node>=_n1)
                arc =_index_layer1+ (node-_n1 + 1) * _n2 - 1;
            else
                arc = (node + 1) * _n_layer1 - 1;




        }

        void nextOut(Arc& arc) const {
            if (arc<_index_layer1 && arc % _n_layer1 == 0) arc = 0;
            if (arc>=_index_layer1 && (arc-_index_layer1) % _n2 == 0) arc = 0;



            --arc;
        }

        void firstIn(Arc& arc, const Node& node) const {
            if (node<_n1)
                arc = -1;
            else
            if (node<_end_layer1)
                arc = _index_layer1 + node - _end_layer1; //to check
            else
                arc = _arc_num + node - _node_num;



        }

        void nextIn(Arc& arc) const {
            if (arc>=_index_layer1)
            {
                arc -= _n2;
                if (arc < _index_layer1) arc = -1;
            }
            else {
                arc -= _n_layer1;

                if (arc < 0) arc = -1;
            }
        }

    };

    /// \ingroup graphs
    ///
    /// \brief A directed full graph class.
    ///
    /// FullTransShipmentgraph is a simple and fast implementation of directed full
    /// (complete) graphs for the Transhipment problem. It contains an arc from each node of the supply to each node of intermediate location, and an arc from each node of the intermediate location to each node of the demands.
    ///
    /// \sa FullTransShipmentgraph
    class FullTransShipmentgraph : public FullTransShipmentgraphBase {
        typedef FullTransShipmentgraphBase Parent;

    public:

        /// \brief Default constructor.
        ///
        /// Default constructor. The number of nodes and arcs will be zero.
        FullTransShipmentgraph() { construct(0,0,0); }

        /// \brief Constructor
        ///
        /// Constructor.
        /// \param n The number of the nodes.
        FullTransShipmentgraph(int n1, int p, int n2) { construct(n1, p, n2); }

        /// \brief Returns the node with the given index.
        ///
        /// Returns the node with the given index. Since this structure is
        /// completely static, the nodes can be indexed with integers from
        /// the range <tt>[0..nodeNum()-1]</tt>.
        /// The index of a node is the same as its ID.
        /// \sa index()
        Node operator()(int ix) const { return Parent::operator()(ix); }

        /// \brief Returns the index of the given node.
        ///
        /// Returns the index of the given node. Since this structure is
        /// completely static, the nodes can be indexed with integers from
        /// the range <tt>[0..nodeNum()-1]</tt>.
        /// The index of a node is the same as its ID.
        /// \sa operator()()
        static int index(const Node& node) { return Parent::index(node); }

        /// \brief Returns the arc connecting the given nodes.
        ///
        /// Returns the arc connecting the given nodes.
        /*Arc arc(Node u, Node v) const {
          return Parent::arc(u, v);
        }*/

        /// \brief Number of nodes.
        int nodeNum() const { return Parent::nodeNum(); }
        /// \brief Number of arcs.
        int64_t arcNum() const { return Parent::arcNum(); }
    };




} //namespace lemon


#endif //LEMON_FULL_GRAPH_H
